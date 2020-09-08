using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;

public class ProjectParser
{
    // sample rate and channels
    private const int SR = 16000, CH = 1;
    // buffer for note_file content
    private string[] fileBuffer;
    // dictionary with metadata
    private Dictionary<string, string> meta = new Dictionary<string, string>();
    // list of dictionaries with singable notes
    private List<Dictionary<string, float>> singables = new List<Dictionary<string, float>>();
    // bpm and song gap
    private float bpm = 0, gap = 0;
    // property for file meta data
    public Dictionary<string, string> Meta { get => meta; set => meta = value; }
    // song as mono float array
    float[] samplesMono;

    // parse note.txt file for pitch and timing information
    public void loadNoteFile(string noteFile)
    {
        fileBuffer = File.ReadAllLines(noteFile);
        foreach (string line in fileBuffer)
        {
            // retrieve metadata
            if (line.StartsWith("#") && singables.Count == 0)
            {
                string[] metaSplit = line.Replace("\n", "").Replace("\r", "").Split(":", 2);
                Meta[metaSplit[0]] = metaSplit[1];
                continue;
            }
            // convert song gap and bpm string to float
            if (bpm == 0)
            {
                bpm = float.Parse(Meta["#BPM"].Replace(".", ","));
                gap = float.Parse(Meta["#GAP"].Replace(".", ","));
            }
            // parse song lines
            if (line.StartsWith(":") || line.StartsWith("*"))
            {
                string[] noteSplit = line.Split(" ");
                int pitch = int.Parse(noteSplit[3]) % 12;
                // start and end beat
                float b_start = float.Parse(noteSplit[1]);
                float b_end = float.Parse(noteSplit[2]);
                // converts beats to song duration
                float t_start = gap + b_start * (15000 / bpm);
                float t_end = gap + (b_start + b_end) * (15000 / bpm);
                // create a dictionary for each singable note
                Dictionary<string, float> singableNote = new Dictionary<string, float>();
                singableNote.Add("t_start", t_start);
                singableNote.Add("t_end", t_end);
                singableNote.Add("pitch", pitch);
                singables.Add(singableNote);
            }
        }
    }

    // load a mono wav file with 16000Hz sample rate
    // ref.: https://www.it-swarm.dev/de/c%23/so-lesen-sie-die-daten-einer-wav-datei-ein-array/940898607/
    public IEnumerable<float[]> readMonoWav16(string wavFile)
    {
        // start with file in byte representation
        byte[] wav = File.ReadAllBytes(wavFile);
        // get past all the other sub chunks to get to the data subchunk:
        // first subchunk id from 12 to 16
        int offset = 12;
        // keep iterating until we find the data chunk (i.e. 64 61 74 61 ...... (i.e. 100 97 116 97 in decimal))
        while(!(wav[offset] == 100 && wav[offset + 1] == 97 && wav[offset + 2] == 116 && wav[offset + 3] == 97))
        {
            offset += 4;
            int chunkSize = wav[offset] + (wav[offset + 1] << 8) + (wav[offset + 2] << 16) + (wav[offset + 3] << 24);
            offset += 4 + chunkSize;
        }
        offset += 8;
        // offset is now positioned to start of actual sound data.
        samplesMono = new float[(wav.Length - offset) / 2];
        // convert to float array
        // 2 bytes per sample (16 bit sound mono)
        for (int i = offset, j = 0; i < wav.Length - offset; i += 2)
        {
            samplesMono[j++] = (float)(((sbyte)wav[i + 1] << 8) | wav[i]);
        }
        // create iterator for audio samples 
        ReadOnlySpan<float> sampleSpan = new ReadOnlySpan<float>(samplesMono);
        foreach (Dictionary<string, float> segment in singables)
        {
            int startSample = (int)(Math.Round((segment["t_start"] * SR) / 1000));
            int endSample = (int)(Math.Round((segment["t_end"] * SR) / 1000));
            yield return samplesMono[startSample..endSample];
        }
    }

    // return a list of all singable pitches
    public List<float> dumpPitches()
    {
        List<float> pitches = new List<float>();
        foreach (Dictionary<string, float> singable in singables)
        {
            pitches.Add(singable["pitch"]);
        }
        return pitches;
    }

    // replace original pitches by updated ones
    public void updatePitches(List<float> newPitches)
    {
        for (int i = 0; i < newPitches.Count; i++)
        {
            singables[i]["pitch"] = newPitches[i];
        }
    }

    // save updated note.txt to disk
    public void saveNoteFile(string noteFile)
    {
        using (StreamWriter file = new StreamWriter(noteFile))
        {
            int i = 0;
            foreach (string line in fileBuffer)
            {
                if (line.StartsWith(":") || line.StartsWith("*"))
                {
                    string[] noteSplit = line.Split(" ");
                    string pitchNew = singables[i]["pitch"].ToString();
                    noteSplit[3] = pitchNew;
                    string lineNew = String.Join(" ", noteSplit);
                    file.WriteLine(lineNew);
                    i++;
                    continue;
                }
                file.WriteLine(line);
            }
        }
    }
}
