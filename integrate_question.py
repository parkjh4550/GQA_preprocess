import json
import config
import re
from nltk.translate.bleu_score import sentence_bleu

ID = False               # get img_id and question_id(GQA)
CAPTION = False         # get caption data
QUESTION = False         # get question data
INTEGRATE = False       # integrate all the information ( caption, question, id etc...)
TRAIN = True            # make train data (keyword, caption, questions)


def load_data_file(dir, file_name, type):
    #########
    # 1. coco caption
    # return dict
    # key : MS COCO img ID,     value : caption

    # 2. GQA question, answer
    # 2-1 question
    # return dict
    # key : GQA img ID,     value : dict
    #                       key : object_name,      value : question
    # 2-2 answer
    # return dict
    # key : GQA img ID,     value : list
    #                       list : answer1, answer2, ....

    # 3. ID load : if there value of "coco" != None, return the id
    # return dict
    # key : MS COCO img ID,     value : GQA img ID

    if type == "caption":
        data = {}
        for name in file_name:
            file_dir = dir + name
            tot_data = load_json(file_dir)
            for img in tot_data['annotations']:
                img_id = img['image_id']  # MS COCO image ID
                caption = img['caption']
                data[img_id] = caption
        return data

    elif type == "question":

        q_data, a_data = {}, {}
        object_dict = load_json("./result/object_id.json")
        question_dict = load_json(config.ques_id_save_dir)

        for name in file_name:
            file_dir = dir + name
            tot_data = load_json(file_dir)

            tot_data_key = list(tot_data.keys())
            for itr, img_key in enumerate(tot_data_key):
                if (itr + 1) % 1000 == 0:
                    print((itr + 1), "  / ", len(tot_data_key), " step preprocessing")
                img = tot_data[img_key]
                img_id = img['imageId']  # GQA image ID

                q_tmp_dict = {}
                a_tmp_list = []
                q_object = []  # objects in the question sentences
                key = img["annotations"]["question"].keys()
                if key:
                    for k in key:
                        object = img["annotations"]["question"][k]
                        q_object.append(object)

                    key = img["annotations"]["fullAnswer"].keys()
                    for k in key:  # compare the objects in the questions and answers
                        object = img["annotations"]["fullAnswer"][k]
                        if object in q_object:
                            name = object_dict[object]
                            tmp_result = {}
                            tmp_entailed = []
                            for entailed in img["entailed"]:
                                sentence = question_dict[entailed]
                                tmp_entailed.append(sentence)
                            tmp_result["entailed"] = tmp_entailed

                            tmp_equivalent = []
                            for equivalent in img["equivalent"]:
                                sentence = question_dict[equivalent]
                                tmp_equivalent.append(sentence)
                            tmp_result["equivalent"] = tmp_equivalent

                            question = img["question"]
                            tmp_result["question"] = question

                            #q_tmp_dict[name] = img["question"]  # key : object, value : questions
                            q_tmp_dict[name] = tmp_result
                            a_tmp_list.append(img["answer"])

                    # add question, answer data to the dictionaries
                    q_data[img_id] = q_tmp_dict
                    a_data[img_id] = a_tmp_list

        return q_data, a_data

    elif type == "img_id":
        # img id dictionary : ms coco, GQA
        img_id_dict = {}
        for name in file_name:
            file_dir = dir + name
            tot_data = load_json(file_dir)

            for k in tot_data.keys():
                img = tot_data[k]
                img_key = re.sub('[^0-9]', '', k)
                if img["coco"] != "None":
                    img_id_dict[img["coco"]] = img_key
        return img_id_dict

    elif type == "ques_id":
        # Question id dictionary : GQA questions' id
        ques_id_dict = {}
        for itr, name in enumerate(file_name):
            print((itr+1), " file processing")
            file_dir = dir + name
            tot_data = load_json(file_dir)

            for num, k in enumerate(list(tot_data.keys())):
                if (num+1)%1000 ==0:
                    print((num+1), ' / ', len(list(tot_data.keys())), ' processing')
                img = tot_data[k]
                ques_id = k
                ques = img["question"]
                ques_id_dict[ques_id] = ques

        return ques_id_dict


def get_paired_question(data):
    result = {}

    for itr, img_id in enumerate(list(data.keys())):
        if (itr+1) % 1000 ==0 :
            print((itr+1), " / ", len(list(data.keys())), " step processing")
        img = data[img_id]
        tmp_dict = {}

        for obj_name in img.keys():
            question = img[obj_name]["question"]
            entailed = img[obj_name]["entailed"]
            if entailed:
                BLEU_list = []
                for sentence in entailed:
                    #BLEUscore = sentence_bleu([reference], hypothesis)
                    BLEUscore = sentence_bleu([question.split()], sentence.split(), weights=(0.5, 0.5))
                    BLEU_list.append(BLEUscore)
                entailed_index = BLEU_list.index(min(BLEU_list))        # find the MOST DIFFERENT question with the reference
                #tmp_dict["question"] = [question, entailed[entailed_index]]
                #tmp_dict["object"] = obj_name
                #tmp_dict["img_id"] = img_id

                if img_id in result.keys():     # if it already have img_id, add the question information to the original data
                    result[img_id][obj_name] = [question, entailed[entailed_index]]
                else:
                    tmp_dict[obj_name] = [question, entailed[entailed_index]]
                    result[img_id] = tmp_dict

                #result.append(tmp_dict)
    return result


def compare_data_coco_gqa(id_data, cap_data, ques_data, ans_data):
    # compare the img id between MS COCO and GQA dataset.
    # if they have same id add data.
    #########
    # return list
    # [ dict1, dict2 , ..... ]
    # key : "coco" ,     value : MS COCO img ID
    # key : "gqa" ,      value : GQA img ID
    # key : "caption",   value : MS COCO caption
    # key : "object",    value : object name
    # key : "question",  value : question sentence
    # Key : "answer" ,   value : answer ( "yes", "no", etc....)

    #수정  필요
    total_data = []
    id_keys = list(id_data.keys())
    ques_id = list(ques_data.keys())
    cap_keys = list(cap_data.keys())
    for itr, cap_id in enumerate(cap_keys):
        if(itr+1)%1000 == 0:
            print((itr+1), " / ",len(cap_keys)," step preprocessing")
            print(" result size : ", len(total_data))
        tmp_list = []
        if cap_id in id_keys:           #check whether the COCO ID matches with GQA ID
            gqa_id = id_data[cap_id]
            if gqa_id in ques_id:       #check whether the gqa_id image has QUESTION data
                if len(list(ques_data[gqa_id].keys())) >= 2:
                    for num, obj in enumerate(list(ques_data[gqa_id].keys())):
                        tmp_dict = {}
                        tmp_dict["coco"] = cap_id
                        tmp_dict["gqa"] = gqa_id
                        tmp_dict["caption"] = cap_data[cap_id]
                        tmp_dict["object"] = obj
                        tmp_dict["question"] = ques_data[gqa_id][obj]
                        tmp_dict["answer"] = ans_data[gqa_id][num]
                        tmp_list.append(tmp_dict)
        total_data.extend(tmp_list)

    return total_data


def get_train_data(total_data):
    # tot_data : list
    # [ dict1, dict2, dict3 , ... ]
    # output: list(caption, sentence) , list(keyword)
    sentences, keywords = [], []

    for data in total_data:
        caption = data["caption"]
        questions = data["question"]
        keyword = data["object"]

        sentences.append(caption)
        for q in questions:
            sentences.append(q)
        keywords.append(keyword)

    return sentences, keywords


def save_json(data, dir):
    with open(dir, "w") as f:
        json.dump(data, f)

def load_json(dir):
    with open(dir, "r") as f:
        data = json.load(f)
    return data


def save_text_file(new_data, old_data, dir, flag):
    if flag == "sentence":
        with open(dir, "w") as f:
            for itr, line in enumerate(new_data):
                f.write(line +'\n')
                if itr%3 == 1 : f.write(line+'\n')
            for line in old_data:
                f.write(line+'\n')

    elif flag == "keyword":
        with open(dir, "w") as f:
            for itr, line in enumerate(new_data):
                f.write(line +'\n')
            for line in old_data:
                f.write(line+'\n')

def load_text_file(dir):
    data = []
    with open(dir, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line: break
            data.append(line[:-1])
    return data

if __name__ == "__main__":

    if ID:
        print("-----ID data preprocessing")
        print("img id processing")
        # 1. img id preprocess
        img_id_data = load_data_file(config.img_id_dir, config.img_id_file, type="img_id")        # (dict) key : mscoco img id, value : gqa_img_id
        save_json(img_id_data, config.img_id_save_dir)
        print("question id processing")
        # 2. question id preprocess
        ques_id_data = load_data_file(config.question_dir, config.question_file, type="ques_id")
        save_json(ques_id_data, config.ques_id_save_dir)
    else:
        print("-----img, question ID data load")
        img_id_data = load_json(config.img_id_save_dir)
        ques_id_data = load_json(config.ques_id_save_dir)

    print("-----complete")

    if CAPTION:
        print("-----MS COCO caption data preprocessing")
        cap_data = load_data_file(config.caption_dir, config.caption_file, type="caption")
        save_json(cap_data, config.caption_save_dir)
    else:
        print("-----MS COCO caption data load")
        cap_data = load_json(config.caption_save_dir)
    print("-----complete")

    if QUESTION:
        print("-----GQA question, object data preprocessing")
        ques_data,ans_data = load_data_file(config.question_dir, config.question_file, type="question")
        save_json(ques_data,config.question_save_dir)
        save_json(ans_data, config.answer_save_dir)

        ques_data = get_paired_question(ques_data)
        save_json(ques_data, config.paired_data_save_dir)
    else:
        print("-----GQA question, object data load")
        ques_data = load_json(config.paired_data_save_dir)
        ans_data = load_json(config.answer_save_dir)
    print("-----complete")

    if INTEGRATE:
        print("----- integrate all the data process")
        total_data = compare_data_coco_gqa(img_id_data, cap_data, ques_data, ans_data)
        save_json(total_data, config.total_data_save_dir)
    else:
        total_data = load_json(config.total_data_save_dir)
    print("-----complete")

    if TRAIN:
        print("----- train the data process")
        train_data, keyword_data = get_train_data(total_data)
        old_train_data = load_text_file(config.old_train_dir)
        old_keyword_data = load_text_file(config.old_keyword_dir)

        # Combine the old and new data, and Save
        save_text_file(train_data, old_train_data, config.train_data_dir, flag="sentence")
        save_text_file(keyword_data, old_keyword_data, config.train_keyword_dir, flag="keyword")
    print("-----complete")