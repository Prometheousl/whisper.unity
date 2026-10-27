using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Diagnostics;
using Whisper.Utils;
// ReSharper disable once RedundantUsingDirective
using System.Linq;

namespace Whisper
{
    /// <summary>Inference metadata only; contains no transcript or microphone audio.</summary>
    public readonly struct WhisperInferenceOutcome
    {
        public readonly int Sequence, InputSamples, PaddingSamples, SegmentCount;
        public readonly double ElapsedMilliseconds;
        public readonly string Status;

        internal WhisperInferenceOutcome(int sequence, int inputSamples, int paddingSamples,
            int segmentCount, double elapsedMilliseconds, string status)
        {
            Sequence = sequence; InputSamples = inputSamples; PaddingSamples = paddingSamples;
            SegmentCount = segmentCount; ElapsedMilliseconds = elapsedMilliseconds; Status = status;
        }
    }

    public delegate void OnStreamResultUpdatedDelegate(string updatedResult);
    public delegate void OnStreamSegmentUpdatedDelegate(WhisperResult segment);   
    public delegate void OnStreamSegmentFinishedDelegate(WhisperResult segment);   
    public delegate void OnStreamFinishedDelegate(string finalResult);
    
    /// <summary>
    /// Parameters of whisper streaming processing.
    /// </summary>
    public class WhisperStreamParams
    {
        /// <summary>
        /// Regular whisper inference params.
        /// </summary>
        public readonly WhisperParams InferenceParam;
        
        /// <summary>
        /// Audio stream frequency. Can't change during transcription.
        /// </summary>
        public readonly int Frequency;
        
        /// <summary>
        /// Audio stream channels count. Can't change during transcription.
        /// </summary>
        public readonly int Channels;

        /// <summary>
        /// Minimal portions of audio that will be processed by whisper stream in seconds.
        /// </summary>
        public readonly float StepSec;

        /// <summary>
        /// Minimal portions of audio that will be processed by whisper in audio samples.    
        /// </summary>
        public readonly int StepSamples;

        /// <summary>
        /// How many seconds of previous segment will be used for current segment.
        /// </summary>
        public readonly float KeepSec;

        /// <summary>
        /// How many samples of previous audio chunk will be used for current chunk.
        /// </summary>
        public readonly int KeepSamples;

        /// <summary>
        /// How many seconds of audio will be recurrently transcribe until context update.
        /// </summary>
        public readonly float LengthSec;
        
        /// <summary>
        /// How many samples of audio will be recurrently transcribe until context update.
        /// </summary>
        public readonly int LengthSamples;

        /// <summary>
        /// Should stream modify whisper prompt for better context handling?
        /// </summary>
        public readonly bool UpdatePrompt;

        /// <summary>
        /// How many recurrent iterations will be used for one chunk?
        /// </summary>
        public readonly int StepsCount;

        /// <summary>
        /// If false stream will use all information from previous iteration.
        /// </summary>
        public readonly bool DropOldBuffer;

        /// <summary>
        /// If true stream will ignore audio chunks with no detected speech.
        /// </summary>
        public readonly bool UseVad;

        public WhisperStreamParams(WhisperParams inferenceParam,
            int frequency, int channels,
            float stepSec = 3f, float keepSec = 0.2f, float lengthSec = 10f,
            bool updatePrompt = true, bool dropOldBuffer = false,
            bool useVad = false)
        {
            InferenceParam = inferenceParam;
            Frequency = frequency;
            Channels = channels;
            
            StepSec = stepSec;
            StepSamples = (int) (StepSec * Frequency * Channels);

            KeepSec = keepSec;
            KeepSamples = (int) (KeepSec * frequency * channels);

            LengthSec = lengthSec;
            LengthSamples = (int) (LengthSec * frequency * channels);

            StepsCount = Math.Max(1, (int) (LengthSec / StepSec) - 1);
            
            UpdatePrompt = updatePrompt;
            DropOldBuffer = dropOldBuffer;
            UseVad = useVad;
        }
    }
    
    /// <summary>
    /// Handling all streaming logic (sliding-window, VAD, etc).
    /// </summary>
    public class WhisperStream
    {
        /// <summary>
        /// Raised when whisper updated stream transcription.
        /// Result contains a full stream transcript from the stream beginning.
        /// </summary>
        public event OnStreamResultUpdatedDelegate OnResultUpdated;
        /// <summary>
        /// Raised when whisper updated current segment transcript.
        /// </summary>
        public event OnStreamSegmentUpdatedDelegate OnSegmentUpdated;
        /// <summary>
        /// Raised when whisper finished current segment transcript. 
        /// </summary>
        public event OnStreamSegmentFinishedDelegate OnSegmentFinished;
        /// <summary>
        /// Raised when whisper finished stream transcription and can start another one.
        /// </summary>
        public event OnStreamFinishedDelegate OnStreamFinished;
        public event Action<WhisperInferenceOutcome> OnInferenceCompleted;
        
        private readonly WhisperStreamParams _param;
        private readonly string _originalPrompt;
        private readonly MicrophoneRecord _microphone;
        
        private readonly List<float> _newBuffer = new List<float>();
        private float[] _oldBuffer = Array.Empty<float>();
        private string _output = "";
        private int _step;
        private bool _isStreaming;
        
        private Task<WhisperResult> _task;
        private Task _updateTask = Task.CompletedTask;
        private bool _processing, _pendingSegmentEnd, _newBufferHasSpeech;
        private WhisperResult _lastResult;
        private int _inferenceSequence;
        private readonly Func<float[], Task<WhisperResult>> _transcribe;

        /// <summary>
        /// Create a new instance of Whisper streaming transcription.
        /// </summary>
        /// <param name="wrapper">Loaded Whisper model which will be used for transcription.</param>
        /// <param name="param">Whisper streaming parameters.</param>
        /// <param name="microphone">Optional microphone input for stream.</param>
        public WhisperStream(WhisperWrapper wrapper, WhisperStreamParams param,
            MicrophoneRecord microphone = null)
            : this(param, microphone, samples => wrapper.GetTextAsync(
                samples, param.Frequency, param.Channels, param.InferenceParam))
        {
        }

        // A single internal seam for deterministic streaming tests without loading a model.
        internal WhisperStream(WhisperStreamParams param, MicrophoneRecord microphone,
            Func<float[], Task<WhisperResult>> transcribe)
        {
            _param = param;
            _originalPrompt = _param.InferenceParam.InitialPrompt;
            _microphone = microphone;
            _transcribe = transcribe;
        }
        
        /// <summary>
        /// Start a new streaming transcription. Must be called before
        /// you start adding new audio chunks.
        /// </summary>
        /// <remarks>
        /// If you set microphone into constructor, it will start listening to it.
        /// Make sure you started microphone by <see cref="MicrophoneRecord.StartRecord"/>.
        /// There is no need to add audio chunks manually using <see cref="AddToStream"/>.
        /// </remarks>
        public void StartStream()
        {
            if (_isStreaming)
            {
                LogUtils.Warning("Stream is already working!");
                return;
            }
            _isStreaming = true;
            
            // if we set microphone - streaming works in auto mode
            if (_microphone != null)
            {
                _microphone.OnChunkReady += MicrophoneOnChunkReady;
                _microphone.OnRecordStop += MicrophoneOnRecordStop;
            }
        }

        /// <summary>
        /// Manually add a new chunk of audio to streaming.
        /// Make sure to call <see cref="StartStream"/> first.
        /// </summary>
        /// <remarks>
        /// If you set microphone into constructor, it will be called automatically.
        /// </remarks>
        public async void AddToStream(AudioChunk chunk)
        {
            if (!_isStreaming)
            {
                LogUtils.Warning("Start streaming first!");
                return;
            }

            if (_param.UseVad)
            {
                if (chunk.IsVoiceDetected)
                {
                    _newBufferHasSpeech = true;
                    _newBuffer.AddRange(chunk.Data);
                    await UpdateSlidingWindow();
                }
                else
                {
                    if (_step <= 0 && _newBuffer.Count == 0 && !_processing)
                    {
                        _oldBuffer = chunk.Data;
                        return;
                    }

                    // Retain one trailing chunk, not a silence backlog while inference is busy.
                    if (!_pendingSegmentEnd) _newBuffer.AddRange(chunk.Data);
                    await UpdateSlidingWindow(true);
                }
            }
            else
            {
                _newBufferHasSpeech = true;
                _newBuffer.AddRange(chunk.Data);
                await UpdateSlidingWindow();
            }
        }

        /// <summary>
        /// Stop current streaming transcription. It will process last
        /// audio chunks and raise <see cref="OnStreamFinished"/> when it's done.
        /// </summary>
        public async void StopStream()
        {
            if (!_isStreaming)
            {
                LogUtils.Warning("Start streaming first!");
                return;
            }
            _isStreaming = false;
            
            // unsubscribe from microphone events for now
            if (_microphone != null)
            {
                _microphone.OnChunkReady -= MicrophoneOnChunkReady;
                _microphone.OnRecordStop -= MicrophoneOnRecordStop;
            }
            
            // finish last part
            await UpdateSlidingWindow(true);
            OnStreamFinished?.Invoke(_output);
            
            // reset stream and drop audio buffer
            Reset();
        }
        
        private Task UpdateSlidingWindow(bool forceSegmentEnd = false)
        {
            _pendingSegmentEnd |= forceSegmentEnd;
            if (_processing) return _updateTask;
            return _updateTask = ProcessSlidingWindow();
        }

        private async Task ProcessSlidingWindow()
        {
            _processing = true;
            try
            {
                while (true)
                {
                    bool finishSegment = _pendingSegmentEnd;
                    int newBufferLen = _newBuffer.Count;
                    if (finishSegment && !_newBufferHasSpeech)
                    {
                        _newBuffer.Clear();
                        _pendingSegmentEnd = false;
                        // Finalize the last real result without transcribing retained silence again.
                        if (_step > 0 && _lastResult != null)
                            FinishSegment(_lastResult, _oldBuffer);
                        return;
                    }
                    if (newBufferLen == 0 || (!finishSegment && newBufferLen < _param.StepSamples))
                        return;

                    int oldBufferLen = _oldBuffer.Length;
                    int nSamplesTake = _param.DropOldBuffer
                        ? Math.Min(oldBufferLen, Math.Max(0,
                            _param.KeepSamples + _param.LengthSamples - newBufferLen))
                        : oldBufferLen;
                    int bufferLen = nSamplesTake + newBufferLen;
                    var buffer = new float[bufferLen];
                    Array.Copy(_oldBuffer, oldBufferLen - nSamplesTake, buffer, 0, nSamplesTake);
                    _newBuffer.CopyTo(0, buffer, nSamplesTake, newBufferLen);
                    _newBuffer.Clear();
                    _newBufferHasSpeech = false;
                    _pendingSegmentEnd = false;

                    WhisperResult res = null;
                    int sequence = ++_inferenceSequence;
                    var stopwatch = OnInferenceCompleted != null ? Stopwatch.StartNew() : null;
                    try
                    {
                        _task = _transcribe(buffer);
                        res = await _task;
                    }
                    finally
                    {
                        if (OnInferenceCompleted != null)
                        {
                            int monoSamples = (int)Math.Round(
                                (double)(bufferLen / _param.Channels) * WhisperWrapper.WhisperSampleRate / _param.Frequency);
                            OnInferenceCompleted(new WhisperInferenceOutcome(sequence, bufferLen,
                                monoSamples > 0 ? Math.Max(0, WhisperWrapper.WhisperSampleRate - monoSamples) : 0,
                                res?.Segments.Count ?? 0, stopwatch?.Elapsed.TotalMilliseconds ?? 0,
                                res == null ? "Failure" : string.IsNullOrWhiteSpace(res.Result) ? "Empty" : "Success"));
                        }
                    }
                    if (res == null)
                    {
                        LogUtils.Error("Whisper inference returned no result.");
                        return;
                    }

                    _lastResult = res;
                    OnSegmentUpdated?.Invoke(res);
                    OnResultUpdated?.Invoke(_output + res.Result);
                    _step++;
                    if (finishSegment || _step >= _param.StepsCount)
                        FinishSegment(res, buffer);
                    else
                        _oldBuffer = buffer;
                    // Drain pending speech/end-of-speech after busy inference even if capture is now silent.
                }
            }
            finally
            {
                _processing = false;
            }
        }

        private void FinishSegment(WhisperResult res, float[] realBuffer)
        {
            _output += res.Result;
            if (_param.UpdatePrompt)
                _param.InferenceParam.InitialPrompt = _originalPrompt + _output;
            int keep = Math.Min(_param.KeepSamples, realBuffer.Length);
            _oldBuffer = new ArraySegment<float>(realBuffer, realBuffer.Length - keep, keep).ToArray();
            _step = 0;
            _lastResult = null;
            OnSegmentFinished?.Invoke(res);
        }
        private void Reset()
        {
            _output = "";
            _step = 0;
            _oldBuffer = Array.Empty<float>();
            _newBuffer.Clear();
            _newBufferHasSpeech = false;
            _pendingSegmentEnd = false;
            _lastResult = null;
        }
        
        private void MicrophoneOnChunkReady(AudioChunk chunk)
        {
            AddToStream(chunk);
        }
        
        private void MicrophoneOnRecordStop(AudioChunk recordedAudio)
        {
            StopStream();
        }

    }
}
