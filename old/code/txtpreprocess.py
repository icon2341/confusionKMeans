import sys
import json
import re
import os
import csv
import math
import deep_disfluency
from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger

def load_json_data(file_path):
    data = json.load(open(file_path))
    return (data["results"]["transcripts"][0]["transcript"], data["results"]["items"])

def identify_disfluency(index, tags):
    tag = tags[index][1]
    if tag == "<f/>":
        return -1
    elif tag[0:2] == "<e":
        return 0
    elif tag[0:3] == "<rm":
        return 1
    elif tag[0:2] == "<i":
        return 2
    elif tag[0:3] == "<rp":
        return 3
    else:
        return 4


def initialize_tagger():
    disf = DeepDisfluencyTagger(
        config_file="../deep_disfluency/experiments/experiment_configs.csv",
        config_number=21,
        saved_model_dir="../deep_disfluency/experiments/021/epoch_40"
        )
    return disf

def tag_text(tagger, long_text):
    disf_tags = []
    words = re.findall(r"[\w']+|[.,!?;]", long_text)

    for word in words:
        tagger.tag_new_word(word)

    for w, t in zip(words, tagger.output_tags):
        disf_tags.append((w, t))
    
    return disf_tags

def utterance_end(token):
    return (token == "." or token == "?")

def get_base_filename(file_name):
    path = os.path.basename(file_name)
    base = os.path.splitext(path)[0]
    return base

def update_sentence_stats(lines, stc_length, speech_rate):
    for line in lines:
        line[5] = stc_length
        line[6] = speech_rate
    return lines

def write_csv(json_file_path, disf, outfile):
    data = load_json_data(json_file_path)
    full_text = data[0]
    words = data[1]

    disf_tags = tag_text(disf, full_text)

    with open(outfile, "w", encoding="UTF8") as f:
        print("writing to " + outfile)
        writer = csv.writer(f)
        writer.writerow(["token", "start_time", "end_time", "is_question", "is_pause", "curr_sentence_length", "speech_rate",
                         "is_edit_word", "is_reparandum", "is_interregnum", "is_repair"])

        stc_word_length = 0
        index = 0
        utterance_start = float(words[0]["start_time"])
        lines = []

        while (index < (len(words)-1)):
            word = words[index]
            content = word["alternatives"][0]["content"]

            while not utterance_end(content) and index < (len(words)-1):
                disf_type = identify_disfluency(index, disf_tags)
                if disf_type == -1:
                    #no disfluency
                    lines.append([content, word.get("start_time"), word.get("end_time"), 0, 0, 0, 0, 0, 0, 0, 0])
                elif disf_type == 0:
                    #edit word
                    lines.append([content, word.get("start_time"), word.get("end_time"), 0, 0, 0, 0, 1, 0, 0, 0])
                elif disf_type == 1:
                    #reparandum
                    lines.append([content, word.get("start_time"), word.get("end_time"), 0, 0, 0, 0, 0, 1, 0, 0])
                elif disf_type == 2:
                    #interregnum
                    lines.append([content, word.get("start_time"), word.get("end_time"), 0, 0, 0, 0, 0, 0, 1, 0])
                elif disf_type == 3:
                    #repair
                    lines.append([content, word.get("start_time"), word.get("end_time"), 0, 0, 0, 0, 0, 0, 0, 1])
                else:
                    print("identify disfluencies returned 4... check tag identification")
                    print(disf_type)
                    print(disf_tags[index])

                if (content == ","):
                        prev_end_time = float(words[index-1].get("end_time"))
                else:
                    prev_end_time = float(word.get("end_time"))
                    stc_word_length += 1

                index += 1
                word = words[index]
                content = word["alternatives"][0]["content"]

                if not utterance_end(content):
                    if (content == ","):
                        next_start_time = float(words[index+1].get("start_time"))
                    else:
                        next_start_time = float(word.get("start_time"))

                    #calculate time between previous and current word, compare with average pause duration = 0.398
                    if ((next_start_time-prev_end_time) >= 0.398):
                        #add line for pause
                        lines.append(["", prev_end_time, next_start_time, 0, 1, 0, 0, 0, 0, 0, 0])
                
            #utterance end reached
            stc_time = float(words[index-1]["end_time"]) - utterance_start
            speech_rate = stc_word_length/stc_time

            final_lines = update_sentence_stats(lines, stc_word_length, speech_rate)
            final_lines.append([content, None, None, 0, 0, stc_word_length, speech_rate, 0, 0, 0, 0])

            if content == "?":
                for line in final_lines:
                    line[3] = 1

            writer.writerows(final_lines)
                
            if (index < (len(words)-2)):
                utterance_start = float(words[index+1]["start_time"])
            
            lines = []
            final_lines = []
            stc_word_length = 0
            index += 1
    print("finished writing " + outfile)
    f.close()
    
def write_dir(directory_name):
    disf = initialize_tagger()

    for (root,dirs,files) in os.walk(directory_name):
        for f in files:
            print(f)
            f_in = os.path.join(root,f)
            f_out = f_in[:-4] + "csv"
            print(f_in)
            if (len(f_in) > 5) and (f_in[-4:] == "json"):
                if not os.path.isfile(f_out):
                    print("calling write csv on " + f_in + " to " + f_out)
                    write_csv(f_in, disf, f_out)



    #for json_file in os.listdir(directory_name):
        #write_csv(json_file, disf)
        #disf.reset()

def smear_csv(csv_file, n_frames, frame_rate=0.04):
    with open(csv_file, "r") as f:
        compressed = []
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            compressed.append(row)
    f.close()
        
    smeared = []
    curr_frame = 0
    for i in range(len(compressed)-1):
        row = compressed[i]
        begin = row[1]
        end = compressed[i+1][1]
        if (begin == "" or end == ""):
            pass
        else:
            num_frames_in_word = math.floor((float(end)-float(begin))/frame_rate)
            for i in range(num_frames_in_word):
                frame = i+curr_frame
                new_row = [frame]
                new_row.extend(row)
                smeared.append(new_row)
            curr_frame += num_frames_in_word
    
    with open("smeared.csv", "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerows(smeared)
    f.close()
        


if __name__ == "__main__":
    write_dir("data/out")


            
        
