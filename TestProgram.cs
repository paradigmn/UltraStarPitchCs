using System.IO;
using System.Collections.Generic;

class TestProgram
{
    // test implementation for the pitch detection pipeline
    static void Main(string[] args)
    {
        // load an ultrastar project (notes & song)
        ProjectParser projectParser = new ProjectParser();
        projectParser.loadNoteFile("/path/to/notes.txt");
        IEnumerable<float[]> audioSegmentsIter = projectParser.readMonoWav16("/path/to/16khz/mono/song.wav");
        // init the audio preprocessor
        AudioPreprocessor audioPreprocessor = new AudioPreprocessor(
            Path.Join("Assets", "Binaries", "Pca", "pcaMeanFp32.npy"),
            Path.Join("Assets", "Binaries", "Pca", "pcaCompFp32.npy"));
        // init the pitch classifier
        PitchClassifier pitchClassifier = new PitchClassifier(
            Path.Join("Assets", "Binaries", "Model", "modelWeights1TransposedFp32.npy"),
            Path.Join("Assets", "Binaries", "Model", "modelWeights2TransposedFp32.npy"),
            Path.Join("Assets", "Binaries", "Model", "modelBias1Fp32.npy"),
            Path.Join("Assets", "Binaries", "Model", "modelBias2Fp32.npy"));
        // iterate over each audio segment
        foreach (float[] segment in audioSegmentsIter)
        {
            // transform segment to a list of features
            float[][] features = audioPreprocessor.transform(segment);
            // determine the most likely pitch for the segment
            int pitch = pitchClassifier.predictBatch(features);
        }
    }
}
