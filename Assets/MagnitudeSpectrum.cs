using System;
using System.Numerics;
using System.Linq;

public class MagnitudeSpectrum
{
    // fft window length
    private readonly int winLen;
    // lookup table for bit reverse indexes
    private readonly int[] idxTable;
    // fft window
    private readonly float[] fftWin;
    // twiddle factors
    private readonly float[] twidRe;
    private readonly float[] twidIm;
	// vector length for simd floating point operations
	private readonly int simdLenFp32 = Vector<float>.Count;

    public MagnitudeSpectrum(int n)
    {
        winLen = n;
		fftWin = hannWindow(winLen);
        buildIdxTable(out idxTable);
        buildtwidFactors(out twidRe, out twidIm);
    }

    // create a lookup table for bit reverse indexes
    private void buildIdxTable(out int[] idxTable)
	{
		idxTable = new int[winLen];
		int bitCnt = (int)Math.Log(winLen, 2) - 1;
		for (int i = 0; i < winLen; i ++)
		{
			int idx = i, cnt = bitCnt;
			for (idxTable[i] = idx & 1, idx >>= 1; idx > 0; idx >>= 1, cnt -= 1)
			{
				idxTable[i] = (idxTable[i] << 1) | (idx & 1);
			}
			idxTable[i] <<= cnt;
		}
	}

    // setup twiddle factors for fft transform
	// ref.: http://www.iowahills.com/Example%20Code/WebFFTCode.txt
    private void buildtwidFactors(out float[] twidRe, out float[] twidIm)
	{
		twidRe = new float[winLen / 2];
		twidIm = new float[winLen / 2];
		float phaseConst = 2.0f * ((float)Math.PI / winLen);
		
		twidRe[0] = 1.0f;
		twidIm[0] = 0.0f;
		twidRe[winLen / 4] = 0.0f;
		twidIm[winLen / 4] = -1.0f;
		twidRe[winLen / 8] = 1 / (float)Math.Sqrt(2);
		twidIm[winLen / 8] = -1 / (float)Math.Sqrt(2);
		twidRe[3 * winLen / 8] = -1 / (float)Math.Sqrt(2);
		twidIm[3 * winLen / 8] = -1 / (float)Math.Sqrt(2);
		for(int i = 1; i < winLen / 8; i++)
		{
			float phi = (float)i * -phaseConst;
			twidRe[i] = (float)Math.Cos(phi);
			twidIm[i] = (float)Math.Sin(phi);
			twidRe[winLen / 4 - i] = -twidIm[i];
			twidIm[winLen / 4 - i] = -twidRe[i];
			twidRe[winLen / 4 + i] = twidIm[i];
			twidIm[winLen / 4 + i] = -twidRe[i];
			twidRe[winLen / 2 - i] = -twidRe[i];
			twidIm[winLen / 2 - i] = twidIm[i];
		}
	}

    // hanning window function
    private float[] hannWindow(int n)
    {
        float[] window = new float[n];
        for (int i = 0; i < n; i++)
        {
            window[i] = 0.5f - 0.5f * (float)Math.Cos(2 * Math.PI * i / n);
        }
        return window;
    }

	private void complexFourier(float[] dataRe, float[] dataIm)
	{
		// rearange data buffer in bit reversed order
		for (int i = 1; i < winLen; i++)
		{
			int j = idxTable[i];
			if (j > i)
			{
				(dataRe[i], dataRe[j]) = (dataRe[j], dataRe[i]);
				(dataIm[i], dataIm[j]) = (dataIm[j], dataIm[i]);
			}
		}

		// calculate first stage (twiddle factor = 1 + 0j)
		for (int i = 0; i < winLen; i += 2)
		{
			float tmpRe = dataRe[i + 1]; 
			float tmpIm = dataIm[i + 1];
			dataRe[i + 1] = dataRe[i] - tmpRe;
			dataIm[i + 1] = dataIm[i] - tmpIm;
			dataRe[i] += tmpRe;
			dataIm[i] += tmpIm;
		}
		// calculate further stages
		int phaseIdx = winLen >> 2;
		for (int i = 2; i < winLen / 2; i *= 2)
		{
			// stage loop
			for (int j = 0; j < winLen; j += (i * 2))
			{
				// block loop
				int twidIdx = 0;
				for (int k = 0; k < i; k++)
				{
					// butterfly loop
					int evenIdx = j + k;
					int oddIdx = evenIdx + i;
					float tmpRe = twidRe[twidIdx] * dataRe[oddIdx] - twidIm[twidIdx] * dataIm[oddIdx]; 
					float tmpIm = twidRe[twidIdx] * dataIm[oddIdx] + twidIm[twidIdx] * dataRe[oddIdx];
					dataRe[oddIdx] = dataRe[evenIdx] - tmpRe;
					dataIm[oddIdx] = dataIm[evenIdx] - tmpIm;
					dataRe[evenIdx] += tmpRe;
					dataIm[evenIdx] += tmpIm;
					twidIdx += phaseIdx;
				}
			}
			phaseIdx >>= 1;
		}
		// calculate final stage with simd intrisics
		Vector<float> vecTmpRe = new Vector<float>();
		Vector<float> vecTmpIm = new Vector<float>();
		for (int i = 0; i < winLen / 2 - simdLenFp32; i += simdLenFp32)
		{
			Vector<float> vecTwidRe = new Vector<float>(twidRe, i);
			Vector<float> vecTwidIm = new Vector<float>(twidIm, i);
			Vector<float> vecDataEvenRe = new Vector<float>(dataRe, i);
			Vector<float> vecDataEvenIm = new Vector<float>(dataIm, i);
			Vector<float> vecDataOddRe = new Vector<float>(dataRe, winLen / 2 + i);
			Vector<float> vecDataOddIm = new Vector<float>(dataIm, winLen / 2 + i);
			vecTmpRe = vecTwidRe * vecDataOddRe - vecTwidIm * vecDataOddIm;
			vecTmpIm = vecTwidRe * vecDataOddIm + vecTwidIm * vecDataOddRe;
			(vecDataEvenRe - vecTmpRe).CopyTo(dataRe, winLen / 2 + i);
			(vecDataEvenIm - vecTmpIm).CopyTo(dataIm, winLen / 2 + i);
			(vecDataEvenRe + vecTmpRe).CopyTo(dataRe, i);
			(vecDataEvenIm + vecTmpIm).CopyTo(dataIm, i);
		}
	}

	// transform a data buffer into its magnitude spectrum
	public void transform(float[] dataRe, int n)
	{
		/* windowing */
		float[] window = fftWin;
		// create custom window for smaller segment processing
		if (n < winLen)
		{
			window = hannWindow(n);
		}
		// multiply data buffer with window
		LinearAlgebraUtils.VecMul(dataRe, window, n, 0);
		// define temporary buffer for complex calculations
		float[] dataIm = new float[winLen];
		// calculate fourier transform
		complexFourier(dataRe, dataIm);
		// get absolute real magnitude spectrum and extrema
		for (int i = 0; i <= winLen / 2 - simdLenFp32; i += simdLenFp32)
		{
			Vector<float> vecDataRe = new Vector<float>(dataRe, i);
			Vector<float> vecDataIm = new Vector<float>(dataIm, i);
			Vector.SquareRoot(vecDataRe * vecDataRe + vecDataIm * vecDataIm).CopyTo(dataRe, i);
		}
		dataRe[winLen / 2] = (float)Math.Sqrt(dataRe[winLen / 2] * dataRe[winLen / 2] + dataIm[winLen / 2] * dataIm[winLen / 2]);
		Array.Clear(dataRe, winLen / 2 + 1, winLen / 2 - 1);
		// scale magnitudes between 0 and 1
		LinearAlgebraUtils.VecMinMax(dataRe, dataRe.Min(), dataRe.Max(), winLen / 2 + 1);
	}

	// uses the complex fft to transform two signals simultaniously
	public void stereoTransform(float[] dataRe, float[] dataIm, int n)
	{
		/* windowing */
		float[] window = fftWin;
		// create custom window for smaller segment processing
		if (n < winLen)
		{
			window = hannWindow(n);
		}
		// multiply data buffer with window
		LinearAlgebraUtils.VecMul(dataRe, window, n, 0);
		LinearAlgebraUtils.VecMul(dataIm, window, n, 0);
		// calculate fourier transform
		complexFourier(dataRe, dataIm);
		// split combined spectrum into magnitude signal spectra
		float aRe = 0, aIm = 0, bRe = 0, bIm = 0;
		float tmpRe = dataRe[winLen / 2 - 1];
		float tmpIm = dataIm[winLen / 2 - 1];
		for (int i = 1; i < winLen / 2; i++)
		{
			aRe = dataRe[i] + dataRe[winLen- i];
			aIm = dataIm[i] - dataIm[winLen - i];
			bRe = dataIm[i] + dataIm[winLen - i];
			bIm = dataRe[winLen - i] - dataRe[i];
			dataRe[i] = (float)Math.Sqrt(aRe * aRe + aIm * aIm);
			dataIm[i] = (float)Math.Sqrt(bRe * bRe + bIm * bIm);
		}
		aRe = dataRe[winLen / 2] + tmpRe;
		aIm = dataIm[winLen / 2] - tmpIm;
		bRe = dataIm[winLen / 2] + tmpIm;
		bIm = tmpRe - dataRe[winLen / 2];
		dataRe[winLen / 2] = (float)Math.Sqrt(aRe * aRe + aIm * aIm);
		dataIm[winLen / 2] = (float)Math.Sqrt(bRe * bRe + bIm * bIm);
		// set unused buffer parts to zero
		Array.Clear(dataRe, winLen / 2 + 1, winLen / 2 - 1);
		Array.Clear(dataIm, winLen / 2 + 1, winLen / 2 - 1);
		// scale magnitudes between 0 and 1
		LinearAlgebraUtils.VecMinMax(dataRe, dataRe.Min(), dataRe.Max(), winLen / 2 + 1);
		LinearAlgebraUtils.VecMinMax(dataIm, dataIm.Min(), dataIm.Max(), winLen / 2 + 1);
	}
}