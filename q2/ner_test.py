import pickle

import numpy as np
import pandas as pd

import spacy

from tqdm import trange
import torch
from torch.optim import Adam
from torch.utils.data import RandomSampler, TensorDataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification

from seqeval.metrics import classification_report, accuracy_score, f1_score

from q2.utils import merge_overlaps

SAVE_PATH = 'saved_bert_model'

CONT_TOKEN = 'X'

BOS_TOKEN = '[CLS]'

EOS_TOKEN = '[SEP]'

nlp = spacy.load("en_core_web_lg")
black_list = nlp.Defaults.prefixes
black_list.append('\\n')
prefix_blacklist_regex = spacy.util.compile_prefix_regex(black_list)
nlp.tokenizer.prefix_search = prefix_blacklist_regex.search

cv_parts = ['Name', 'College Name', 'Degree', 'Graduation Year', 'Years of Experience', 'Companies worked at', 'Designation', 'Skills', 'Location', 'Email Address']
cv_parts_dict = {x: x.replace(' ', '_') for x in cv_parts}

df = pd.read_json('hackaton_data/train.json', lines=True).drop('extras', axis=1)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
def prepare_labels(df):
    crawled_cv_parts = []
    for i in range(df.index.size):
        crawled_cv_part = []
        for annotation in df['annotation'][i]:
            if annotation.get('label'):
                cv_part = cv_parts_dict.get(annotation.get('label')[0])
                b_idx = annotation.get('points')[0].get('start')
                e_idx = annotation.get('points')[0].get('end') + 1
                crawled_cv_part.append((b_idx, e_idx, cv_part))

        crawled_cv_parts.append(merge_overlaps(crawled_cv_part))
    return crawled_cv_parts


def split_cv_into_sentences(df):
    cv_part_tokens = []
    sentences = []

    for i in range(df.index.size):
        row = df.iloc[i]
        content, cv_part_values = row['content'], row['cv_parts']
        doc = nlp(content)
        cv_part_token = spacy.training.offsets_to_biluo_tags(doc, cv_part_values)
        doc_df = pd.DataFrame([list(doc), cv_part_token]).T
        start_end_indices = []
        for i in range(doc_df.index.size):
            if doc_df[0][i].text is '.' and doc_df[1][i] is 'O':
                start_end_indices.append(i)
        start_end_indices.append(len(doc))
        last_index = 0
        splitted_part = []
        for pos in start_end_indices:
            splitted_part.append([list(doc)[last_index:pos], cv_part_token[last_index:pos]])
            last_index = pos

        for d in splitted_part:
            cv_part_token = []
            for cpt in d[1]:
                if cpt == '-':
                    cv_part_token.append('O')
                else:
                    cv_part_token.append(cpt)

            if len(set(cv_part_token)) > 1:
                sentences.append(d[0])
                cv_part_tokens.append(cv_part_token)
    return (cv_part_tokens, sentences)


def build_cv_part_token_vocab(cv_part_tokens):
    cv_part_token_distinct_values = set([CONT_TOKEN, BOS_TOKEN, EOS_TOKEN])
    cv_part_token_distinct_values.update([y for x in cv_part_tokens for y in x])

    tag_idx_dict = {y: x for x, y in enumerate(cv_part_token_distinct_values)}
    idx_tag_dict = {y: x for x, y in tag_idx_dict.items()}

    return tag_idx_dict, idx_tag_dict

def build_dataset(sentences, cv_part_tokens):
    tokenized_texts = []
    word_piece_labels = []
    for word_list, label in zip(sentences, cv_part_tokens):
        temp_lable = [BOS_TOKEN]
        temp_token = [BOS_TOKEN]

        for word, lab in zip(word_list, label):
            token_list = bert_tokenizer.tokenize(word.text)
            for t_i, token in enumerate(token_list):
                temp_token.append(token)
                if t_i == 0:
                    temp_lable.append(lab)
                else:
                    temp_lable.append(CONT_TOKEN)

        temp_lable.append(EOS_TOKEN)
        temp_token.append(EOS_TOKEN)

        tokenized_texts.append(temp_token)
        word_piece_labels.append(temp_lable)

    padded_x = pad_sequences([bert_tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                             maxlen=512, dtype="long", truncating="post", padding="post")

    padded_y = pad_sequences([[tag_idx_dict.get(l) for l in label] for label in word_piece_labels], maxlen=512, value=tag_idx_dict["O"], padding="post", dtype="long", truncating="post")

    return padded_x, padded_y

def build_data_generator(X_all, Y_all):
    x_train, x_test, y_train, y_test = train_test_split(X_all, Y_all, random_state=4, test_size=0.25)

    x_train = torch.tensor(x_train)
    x_test = torch.tensor(x_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    dataset_train = TensorDataset(x_train, y_train)
    batch_loader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=3)

    dataset_validation = TensorDataset(x_test, y_test)
    batch_loader_test = DataLoader(dataset_validation, sampler=RandomSampler(dataset_validation), batch_size=3)

    return batch_loader_train, batch_loader_test

def build_model(tag_idx_dict, load_pretrained=False):
    model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag_idx_dict)).cuda()
    if load_pretrained:
        model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

    parameters_to_finetune = list(model.named_parameters())

    optimizer = Adam([{'params': [p for n, p in parameters_to_finetune if not any(nd in n for nd in ['bias', 'gamma', 'beta'])],
                       'weight_decay_rate': 0.01}], lr=1e-4)

    return model, optimizer

def train_model(model, epoch_count, batch_loader_train):
    for x in trange(epoch_count, desc="Epoch"):
        model.train()
        train_loss = 0
        train_samples, train_steps = 0, 0
        for step, batch in enumerate(batch_loader_train):
            batch = tuple(t.cuda() for t in batch)
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.type(torch.LongTensor)
            batch_labels = batch_labels.type(torch.LongTensor)
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.cuda()

            loss = model(batch_inputs, token_type_ids=None, labels=batch_labels)
            loss.backward()
            train_loss += loss.item()
            train_samples += batch_inputs.size(0)
            train_steps += 1
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            model.zero_grad()


def evaluate_test_cases(model, batch_loader_test):
    y_true = []
    y_pred = []

    for batch in batch_loader_test:
        padded_x, label_ids = batch

        padded_x = padded_x.type(torch.LongTensor)
        label_ids = label_ids.type(torch.LongTensor)

        padded_x = padded_x.cuda()
        label_ids = label_ids.cuda()

        with torch.no_grad():
            logits = model(padded_x, token_type_ids=None)

        logits = logits.detach().cpu().numpy()
        logits = [list(p) for p in np.argmax(logits, axis=2)]

        label_ids = label_ids.to('cpu').numpy()

        for i, token_idx in enumerate(padded_x):
            y_true_list, y_pred_list = [], []

            for j, t_i in enumerate(token_idx):
                if t_i == 0:
                    break
                if idx_tag_dict[label_ids[i][j]] != "X" and idx_tag_dict[label_ids[i][j]] != "[CLS]" and idx_tag_dict[label_ids[i][j]] != "[SEP]":
                    y_true_list.append(idx_tag_dict[label_ids[i][j]])
                    y_pred_list.append(idx_tag_dict[logits[i][j]])

            y_true.append(y_true_list)
            y_pred.append(y_pred_list)

    print(classification_report(y_true, y_pred, digits=5))
    print("Accuracy: %f" % (accuracy_score(y_true, y_pred)))
    print("F1: %f" % (f1_score(y_true, y_pred)))

def generate_cv_summary(single_cv_json):
    df = pd.read_json(single_cv_json, lines=True)
    df['cv_parts'] = prepare_labels(df)
    for i in range(df.index.size-1):
        kkk = df[i:i+1]

        kkk_cv_part_tokens = []
        kkk_sentences = []

        for i in kkk.index:
            content = kkk['content'][i]
            cv_part_values = kkk['cv_parts'][i]

            doc = nlp(content)
            cv_part_token = spacy.training.offsets_to_biluo_tags(doc, cv_part_values)
            doc_df = pd.DataFrame([list(doc), cv_part_token]).T
            start_end_indices = []
            for i in range(len(doc_df)):
                if doc_df[0][i].text is '.' and doc_df[1][i] is 'O':
                    start_end_indices.append(i)
            start_end_indices.append(len(doc))

            last_index = 0
            splitted_part = []
            for pos in start_end_indices:
                splitted_part.append([list(doc)[last_index:pos], cv_part_token[last_index:pos]])
                last_index = pos

            for d in splitted_part:
                cv_part_token = []
                for cpt in d[1]:
                    if cpt == '-':
                        cv_part_token.append('O')
                    else:
                        cv_part_token.append(cpt)

                if len(set(cv_part_token)) > 1:
                    kkk_sentences.append(d[0])
                    kkk_cv_part_tokens.append(cv_part_token)


        kkk_tokenized_texts = []
        kkk_word_piece_labels = []

        for word_list, label in zip(kkk_sentences, kkk_cv_part_tokens):

            # Add [CLS] at the front
            temp_lable = [BOS_TOKEN]
            temp_token = [BOS_TOKEN]

            for word, lab in zip(word_list, label):
                token_list = bert_tokenizer.tokenize(word.text)
                for t_i, token in enumerate(token_list):
                    temp_token.append(token)
                    if t_i == 0:
                        temp_lable.append(lab)
                    else:
                        temp_lable.append(CONT_TOKEN)

                        # Add [SEP] at the end
            temp_lable.append(EOS_TOKEN)
            temp_token.append(EOS_TOKEN)

            kkk_tokenized_texts.append(temp_token)
            kkk_word_piece_labels.append(temp_lable)


        kkk_input_ids = pad_sequences([bert_tokenizer.convert_tokens_to_ids(txt) for txt in kkk_tokenized_texts],
                                      maxlen=512, dtype="long", truncating="post", padding="post")

        kkk_tags = pad_sequences([[tag_idx_dict.get(l) for l in lab] for lab in kkk_word_piece_labels], maxlen=512, value=tag_idx_dict["O"],
                                 padding="post", dtype="long", truncating="post")


        kkk_input_ids = torch.LongTensor(kkk_input_ids)
        kkk_input_ids = kkk_input_ids.cuda()

        with torch.no_grad():
            logits = model(kkk_input_ids, token_type_ids=None, )

        logits = logits.detach().cpu().numpy()
        logits = [list(p) for p in np.argmax(logits, axis=2)]

        retrieved_values = []
        for i, logit in enumerate(logits):
            for j, idx in enumerate(logit):
                retrieved_tag = idx_tag_dict.get(idx)
                retrieved_token = bert_tokenizer.convert_ids_to_tokens([kkk_input_ids[i][j].cpu().item()])[0]
                if retrieved_tag not in [CONT_TOKEN, 'O', BOS_TOKEN] and retrieved_token != '[PAD]' and j <= len(kkk_sentences[i]):
                    my_pos = j + 1
                    while my_pos < len(kkk_word_piece_labels[i]) and kkk_word_piece_labels[i][my_pos] == CONT_TOKEN:
                        retrieved_token += bert_tokenizer.convert_ids_to_tokens([kkk_input_ids[i][my_pos].cpu().item()])[0].lstrip('#')
                        my_pos += 1

                    retrieved_values.append((retrieved_tag, retrieved_token))


        grouped_entities = {}
        doc_df = ""
        last_tag = None
        for i, (tag, token) in enumerate(retrieved_values):
            if last_tag is None:
                last_tag = tag

            if doc_df and tag.startswith('B-'):
                grouped_entities.setdefault(last_tag[2:], []).append(doc_df)
                doc_df = '\n' + token
            elif doc_df and tag[2:] == retrieved_values[i - 1][0][2:]:
                doc_df += ' ' + token
            elif doc_df == '':
                doc_df = '\n' + token
            else:
                grouped_entities.setdefault(last_tag[2:], []).append(doc_df)
                doc_df = '\n' + token

            last_tag = tag

        for cv_part in cv_parts:
            if cv_part.replace(' ', '_') in grouped_entities:
                print('\n'+cv_part)
                print('-'*len(cv_part),"".join(grouped_entities.get(cv_part.replace(' ', '_'))))



df['cv_parts'] = prepare_labels(df)
cv_part_tokens, sentences = split_cv_into_sentences(df)
tag_idx_dict, idx_tag_dict = build_cv_part_token_vocab(cv_part_tokens)
X_all, Y_all = build_dataset(sentences, cv_part_tokens)
batch_loader_train, batch_loader_test = build_data_generator(X_all, Y_all)

model, optimizer = build_model(tag_idx_dict)
train_model(model, 10, batch_loader_train)
evaluate_test_cases(model, batch_loader_test)


# save to file
torch.save(model.state_dict(), SAVE_PATH)
token_info_dict = {'tag_idx_dict': tag_idx_dict, 'idx_tag_dict': idx_tag_dict}
with open('token_info.pkl', 'wb') as o:
    pickle.dump(token_info_dict, o)


# load from saved file
with open('token_info.pkl', 'rb') as i:
    token_info_dict = pickle.load(i)
tag_idx_dict = token_info_dict.get('tag_idx_dict')
idx_tag_dict = token_info_dict.get('idx_tag_dict')

model, optimizer = build_model(tag_idx_dict, load_pretrained=True)

single_cv_json = 'hackaton_data/test.json'
generate_cv_summary(single_cv_json)
