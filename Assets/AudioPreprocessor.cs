using System;


public class AudioPreprocessor
{
    // audio sample rate for processing
    private const int SampleRate = 16000;
    // length of fft transformation window (power of two)
    private const int WinLen = 2048;
    // step size for sliding fft window
    private const int Stride = 128;
    // constants for pca transformation
    private readonly float[] pcaMean;
    private readonly int[] pcaMeanDims;
    private readonly float[] pcaComp;
    private readonly int[] pcaCompDims;
    // object for fourier magnitude transform
    private readonly MagnitudeSpectrum fftMagnitudeTrafo;

    public AudioPreprocessor(string pcaMeanFile, string pcaCompFile)
    {
        // load pca mean vector
        LinearAlgebraUtils.loadNdarray(pcaMeanFile, out pcaMean, out pcaMeanDims);
        // load pca component matrix
        LinearAlgebraUtils.loadNdarray(pcaCompFile, out pcaComp, out pcaCompDims);
        // init fft magnitude spectrum transformer
        fftMagnitudeTrafo = new MagnitudeSpectrum(WinLen);
    }

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
        var watch = System.Diagnostics.Stopwatch.StartNew();
        // init processing buffer
        float[] procBuffer = new float[WinLen];
        // output matrix
        float[][] features;
        // process small segments differently
        if (audioSegment.Length < WinLen)
        {
            features = new float[pcaCompDims[0]][];
            // copy audio segment into buffer
            Buffer.BlockCopy(audioSegment, 0, procBuffer, 0, audioSegment.Length * 4);
            // fourier transform frame
            fftMagnitudeTrafo.transform(procBuffer, audioSegment.Length);
            // apply pca on fft frame -> generate feature row
            features[0] = framePca(procBuffer);
        }
        else
        {
            // calculate number of frames
            int frameCount = (audioSegment.Length - WinLen) / Stride + 1;
            features = new float[frameCount * pcaCompDims[0]][];

            // iterate over each frame
            for (int frameIdx = 0; frameIdx < frameCount; frameIdx++)
            {
                // copy audio segment into buffer
                Buffer.BlockCopy(audioSegment, frameIdx * Stride, procBuffer, 0, WinLen * 4);
                // fourier transform frame
                fftMagnitudeTrafo.transform(procBuffer, WinLen);
                // apply pca on fft frame -> generate feature row
                features[frameIdx] = framePca(procBuffer);
            }
        }
        watch.Stop();
        Console.WriteLine(watch.ElapsedMilliseconds);
        return features;
    }
}
