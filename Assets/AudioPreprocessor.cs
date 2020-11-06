using System;


public class AudioPreprocessor
{
    // audio sample rate for processing
    private const int SampleRate = 16000;
    // length of fft transformation window (power of two)
    private const int WinLen = 2048;
    // step size for sliding fft window
    private const int BaseStride = 128;
    // constants for pca transformation
    private readonly float[] pcaMean, pcaComp;
    private readonly int[] pcaMeanDims, pcaCompDims; 
    // object for fourier magnitude transform
    private readonly MagnitudeSpectrum magnitudeSpectrum;

    public AudioPreprocessor(string pcaMeanFile, string pcaCompFile)
    {
        // load pca mean vector
        LinearAlgebraUtils.loadNdarray(pcaMeanFile, out pcaMean, out pcaMeanDims);
        // load pca component matrix
        LinearAlgebraUtils.loadNdarray(pcaCompFile, out pcaComp, out pcaCompDims);
        // init magnitude spectrum transformer
        magnitudeSpectrum = new MagnitudeSpectrum(WinLen);
    }

    // reduce feature size by applying a pca
    private float[] framePca(float[] procBuffer)
    {
        // substract pca mean values from data
        LinearAlgebraUtils.VecSub(procBuffer, pcaMean, pcaMeanDims[0]);
        // dot multiply transposed component matrix with data vector (comp * vec.T)
        return LinearAlgebraUtils.MatVecDot(pcaComp, procBuffer, pcaCompDims);
    }

    // transforms an audio segment into a feature matrix
    public float[][] transform(float[] audioSegment)
    {
        // init processing buffer
        float[] procBuffer = new float[WinLen];
        float[] procBuffer2 = new float[WinLen];
        // output matrix
        float[][] features;
        // process small segments differently
        if (audioSegment.Length < WinLen)
        {
            features = new float[1][];
            // copy audio segment into buffer
            Buffer.BlockCopy(audioSegment, 0, procBuffer, 0, audioSegment.Length * 4);
            // fourier transform frame
            magnitudeSpectrum.transform(procBuffer, audioSegment.Length);
            // apply pca on fft frame -> generate feature row
            features[0] = framePca(procBuffer);
        }
        else
        {
            // use dynamic stride for larger inputs
            int stride = BaseStride * (audioSegment.Length / WinLen);
            // calculate number of frames
            int frameCount = (audioSegment.Length - WinLen) / stride + 1;
            features = new float[frameCount][];
            int frameIdx = 0;
            // use normal fft for first frame if count is uneven
            if ((frameCount & 1) != 0)
            {
                // copy audio segment into buffer
                Buffer.BlockCopy(audioSegment, frameIdx++ * stride * 4, procBuffer, 0, WinLen * 4);
                // fourier transform frame
                magnitudeSpectrum.transform(procBuffer, WinLen);
                // apply pca on fft frame -> generate feature row
                features[0] = framePca(procBuffer);
            }
            // use stereo fft for further frames
            for (;frameIdx < frameCount; frameIdx += 2)
            {
                // copy audio segment into buffer
                Buffer.BlockCopy(audioSegment, frameIdx * stride * 4, procBuffer, 0, WinLen * 4);
                Buffer.BlockCopy(audioSegment, (frameIdx + 1) * stride * 4, procBuffer2, 0, WinLen * 4);
                // fourier transform frame
                magnitudeSpectrum.stereoTransform(procBuffer, procBuffer2, WinLen);
                // apply pca on fft frame -> generate feature row
                features[frameIdx] = framePca(procBuffer);
                features[frameIdx + 1] = framePca(procBuffer2);
            }
        }
        return features;
    }
}
