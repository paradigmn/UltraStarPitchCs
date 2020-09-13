using System;
using System.Linq;

public class PitchClassifier
{
    private readonly float[] w1, w2;
    private readonly int[] w1Dims, w2Dims;
    private readonly float[] b1, b2;
    private readonly int[] b1Dims, b2Dims;

    private PitchClassifier(string w1File, string w2File, string b1File, string b2File)
    {
        // load weights and biases from files
        LinearAlgebraUtils.loadNdarray(w1File, out w1, out w1Dims);
        LinearAlgebraUtils.loadNdarray(w2File, out w2, out w2Dims);
        LinearAlgebraUtils.loadNdarray(b1File, out b1, out b1Dims);
        LinearAlgebraUtils.loadNdarray(b2File, out b2, out b2Dims);
    }

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

    public float[] predict(float[] features)
    {
        float[] l1 = LinearAlgebraUtils.MatVecDot(w1, features, w1Dims);
        LinearAlgebraUtils.VecAdd(l1, b1, b1Dims[0]);
        LinearAlgebraUtils.VecReLu(l1, w1Dims[1]);
        float[] l2 = LinearAlgebraUtils.MatVecDot(w2, l1, w2Dims);
        LinearAlgebraUtils.VecAdd(l2, b2, b2Dims[0]);
        return softMax(l2, w2Dims[1]);
    }
}