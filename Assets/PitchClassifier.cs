using System;
using System.Linq;

public class PitchClassifier
{
    private readonly float[] w1, w2;
    private readonly int[] w1Dims, w2Dims;
    private readonly float[] b1, b2;
    private readonly int[] b1Dims, b2Dims;

    public PitchClassifier(string w1File, string w2File, string b1File, string b2File)
    {
        // load weights and biases from files
        LinearAlgebraUtils.loadNdarray(w1File, out w1, out w1Dims);
        LinearAlgebraUtils.loadNdarray(w2File, out w2, out w2Dims);
        LinearAlgebraUtils.loadNdarray(b1File, out b1, out b1Dims);
        LinearAlgebraUtils.loadNdarray(b2File, out b2, out b2Dims);
    }

    // numeric stable implementation of the softmax function
    private float[] softMax(float[] output, int n)
    {
        double sum = 0;
        double offset = output.Max();
        for (int i = 0; i < n; i++)
        {
            sum += Math.Exp(output[i] - offset);
        }
        double scale = offset + Math.Log(sum);
        for (int i = 0; i < n; i++)
        {
            output[i] = (float)Math.Exp(output[i] - scale);
        }
        return output;
    }

    // get index of the array maximum
    private int argMax(float[] data)
    {
        return data.ToList().IndexOf(data.Max());
    }

    // add two arrays element wise
    private void arrAdd(float[] a, float[] b)
    {
        for (int i = 0; i < a.Length; i++)
        {
            a[i] += b[i];
        }
    }

    // get the most likely pitch for the feature array
    public int predict(float[] features)
    {
        return argMax(predictProb(features));
    }

    // get the pitch probabilities for the feature array
    public float[] predictProb(float[] features)
    {
        float[] l1 = LinearAlgebraUtils.MatVecDot(w1, features, w1Dims);
        LinearAlgebraUtils.VecAdd(l1, b1, b1Dims[0]);
        LinearAlgebraUtils.VecReLu(l1, w1Dims[0]);
        float[] l2 = LinearAlgebraUtils.MatVecDot(w2, l1, w2Dims);
        LinearAlgebraUtils.VecAdd(l2, b2, b2Dims[0]);
        return softMax(l2, w2Dims[0]);
    }

    // averages batch probabilities and returns the most likely pitch
    public int predictBatch(float[][] featureBatch)
    {
        float[] tmp = new float[w2Dims[0]];
        foreach (float[] features in featureBatch)
        {
            arrAdd(tmp, predictProb(features));
        }
        return argMax(tmp);
    }

    // get an array of pitch probabilities for the feature array batch  
    public float[][] predictBatchProb(float[][] featureBatch)
    {
        float[][] tmp = new float[w2Dims[0]][];
        foreach (float[] features in featureBatch)
        {
            tmp.Append(predictProb(features));
        }
        return tmp;
    }
}