using System;
using System.Linq;
using System.Numerics;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Collections.Concurrent;

class Program
{
    static void Main(string[] args)
    {
        ProjectParser notes = new ProjectParser();
        notes.loadNoteFile("/home/ruben/Testing/notes.txt");
        // notes.loadNoteFile("/home/ruben/Testing/edge/notes.txt");
        IEnumerable<float[]> audioSegmentsIter = notes.readMonoWav16("/home/ruben/Testing/song.wav");
        // IEnumerable audioSegmentsIter = notes.readMonoWav16("/home/ruben/Testing/edge/song.wav");

        List<float> pitchesOld = notes.dumpPitches();

        // notes.updatePitches(test);
        // notes.saveNoteFile("/home/ruben/Testing/test.txt");

        AudioPreprocessor preproc = new AudioPreprocessor(
            "/home/ruben/Workspace_home/UltraStarPitchCs/Assets/Binaries/Pca/pcaMeanFp32.npy", 
            "/home/ruben/Workspace_home/UltraStarPitchCs/Assets/Binaries/Pca/pcaCompFp32.npy");

        var watch = System.Diagnostics.Stopwatch.StartNew();
        //Parallel.ForEach(audioSegmentsIter, segment =>
        foreach (float[] segment in audioSegmentsIter)
        {
            float[][] features = preproc.transform(segment);
            //Console.WriteLine("");
        }

        watch.Stop();
        Console.WriteLine(watch.ElapsedMilliseconds);

        // Random rand = new Random(); 
        // float[] test = new float[2048];
        // for (int i = 0; i < 2048; i++)
        // {
        //     test[i] = (float)rand.NextDouble();
        // }

        // watch = System.Diagnostics.Stopwatch.StartNew();
        // float test2 = test.Max();
        // watch.Stop();
        // Console.WriteLine(watch.ElapsedTicks);
        // watch = System.Diagnostics.Stopwatch.StartNew();
        // float test1 = LinearAlgebraUtils.VecMaximum(test);
        // watch.Stop();
        // Console.WriteLine(watch.ElapsedTicks);

        Console.WriteLine("blub");
    }
}
