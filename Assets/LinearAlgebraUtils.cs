using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Text.RegularExpressions;

public static class LinearAlgebraUtils
{
    // get simd vector length
    private static int simdLenFp32 = Vector<float>.Count;

    // a minimal parser for numpys .npy files (numeric data container)
    // this implementation just expects arrays with dtype float in c contiguous order
    // only a 1d array is returned, since multi dimension arrays are painfully slow in c#
    public static void loadNdarray(string numpyFile, out float[] ndarray, out int[] dims)
    {
        // start with file in byte representation
        byte[] npy = File.ReadAllBytes(numpyFile);
        // skip file preamble (8 bytes)
        int offset = 8;
        // get header length (2 bytes)
        int headerLen = npy[offset] + npy[offset + 1] * 256;
        // add header lenght bytes to offset 
        offset += 2;
        // get the header string
        string header = System.Text.Encoding.UTF8.GetString(npy, offset, headerLen);
        // get shape information from header with regex
        // note: should probably be replaced with a proper json parser!
        string shapePattern = @"(?<='shape': \()(.*?)(?=\))";
        string shape = Regex.Match(header, shapePattern).Value;
        dims = shape.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(Int32.Parse).ToArray();
        // add header lenght to offset
        offset += headerLen;
        // get absolute lenght of ndarray
        int ndarrayLen = 1;
        for (int i = 0; i < dims.Length; i++)
        {
            ndarrayLen *= dims[i];
        }    
        // convert subsequent bytes to double array (4 byte == 1 float)
        ndarray = new float[ndarrayLen];
        Buffer.BlockCopy(npy, offset, ndarray, 0, ndarrayLen * 4);
    }

    // element wise addition (a += b) for n components
    public static void VecAdd(float[] a, float[] b, int n)
    {
        int i;
        for (i = 0; i < n - simdLenFp32; i += simdLenFp32)
        {
            // run simd vectors addition
            Vector<float> vecA = new Vector<float>(a, i);
            Vector<float> vecB = new Vector<float>(b, i);
            Vector.Add(vecA, vecB).CopyTo(a, i);
        }
        for (; i < n; i++)  
        {
            // run normal addition for remaining values
            a[i] += b[i];
        }  
    }

    // element wise substaction (a -= b) for n components
    public static void VecSub(float[] a, float[] b, int n)
    {
        int i;
        for (i = 0; i < n - simdLenFp32; i += simdLenFp32)
        {
            // run simd vectors subtraction
            Vector<float> vecA = new Vector<float>(a, i);
            Vector<float> vecB = new Vector<float>(b, i);
            Vector.Subtract(vecA, vecB).CopyTo(a, i);
        }
        for (; i < n; i++)  
        {
            // run normal substraction for remaining values
            a[i] -= b[i];
        }  
    }

    // element wise multiplication (a *= b) for n components
    public static void VecMul(float[] a, float[] b, int n)
    {
        int i;
        for (i = 0; i < n - simdLenFp32; i += simdLenFp32)
        {
            // run simd vectors multiplication
            Vector<float> vecA = new Vector<float>(a, i);
            Vector<float> vecB = new Vector<float>(b, i);
            Vector.Multiply(vecA, vecB).CopyTo(a, i);
        }
        for (; i < n; i++)  
        {
            // run normal multiplication for remaining values
            a[i] *= b[i];
        }  
    }

    // element wise multiplication (a *= b) for n components with offset
    public static void VecMul(float[] a, float[] b, int n, int aOffset)
    {
        int i;
        for (i = 0; i < n - simdLenFp32; i += simdLenFp32)
        {
            // run simd vectors multiplication
            Vector<float> vecA = new Vector<float>(a, aOffset + i);
            Vector<float> vecB = new Vector<float>(b, i);
            Vector.Multiply(vecA, vecB).CopyTo(a, i);
        }
        for (; i < n; i++)  
        {
            // run normal multiplication for remaining values
            a[aOffset + i] *= b[i];
        }  
    }

    public static void VecMul(ReadOnlySpan<float> a, float[] b, float[] output, int n)
    {
        int i;
        for (i = 0; i < n - simdLenFp32; i += simdLenFp32)
        {
            // run simd vectors multiplication
            Vector<float> vecA = new Vector<float>(a.Slice(i));
            Vector<float> vecB = new Vector<float>(b, i);
            Vector.Multiply(vecA, vecB).CopyTo(output, i);
        }
        for (; i < n; i++)  
        {
            // run normal multiplication for remaining values
            output[i] *= b[i];
        }  
    }

    // dot product of two arrays with length n
    public static float VecDot(float[] a, float[] b, int n, int aOffset)
    {
        int i;
        float sum = 0;
        for (i = 0; i < n - simdLenFp32; i += simdLenFp32)
        {
            // run simd dot product
            Vector<float> vecA = new Vector<float>(a, aOffset + i);
            Vector<float> vecB = new Vector<float>(b, i);
            sum += Vector.Dot(vecA, vecB);
        }
        for (; i < n; i++)
        {
            // run normal dot product for remaining elements
            sum += (a[aOffset + i] * b[i]);
        }
        return sum;
    }

    // element wise min-max-scaling (a = (b - min) / (max - min)) for n components
    public static void VecMinMax(float[] v, float min, float max, int n)
    {
        Vector<float> vecMin = new Vector<float>(min);
        Vector<float> vecMax = new Vector<float>(max) - vecMin;
        int i;
        for (i = 0; i < n - simdLenFp32; i += simdLenFp32)
        {
            // run simd min-max-scaling
            Vector<float> vecV = new Vector<float>(v, i);
            Vector.Divide(vecV - vecMin, vecMax).CopyTo(v, i);
        }
        for (; i < n; i++)
        {
            v[i] = (v[i] - min) / max;
        }
    }

    // matrix vector dot product
    public static float[] MatVecDot(float[] m, float[] v, int[] matDims)
    {
        float[] vOut = new float[matDims[0]];
        int colOffset = 0;
        for (int i = 0; i < matDims[0]; i++)
        {
            vOut[i] = VecDot(m, v, matDims[1], colOffset);
            colOffset += matDims[1];
        }
        return vOut;
    }

    // elementwise relu function
    public static void VecReLu(float[] v, int n)
    {
        int i;
        Vector<float> vecZ = new Vector<float>(0);
        for (i = 0; i < n - simdLenFp32; i += simdLenFp32)
        {
            // run simd vectors multiplication
            Vector<float> vecV = new Vector<float>(v, i);
            Vector.Max(vecZ, vecV).CopyTo(v, i);
        }
        for (; i < n; i++)
        {
            v[i] = Math.Max(0f, v[i]);
        }
    }
}