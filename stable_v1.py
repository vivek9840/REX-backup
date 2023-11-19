import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QProgressBar, QPushButton, QVBoxLayout, QWidget, QSpinBox
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5 import QtCore
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pypdfium2 as pdfium
import re
from spacy.lang.en import English
from scipy.signal import argrelextrema
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from pathlib import Path
from unstructured.cleaners.core import group_broken_paragraphs, replace_unicode_quotes, clean, clean_non_ascii_chars, remove_punctuation

# bytes_string_to_string
def remove_unnecessary_tokens(token):
    token = re.sub(r"[^0-9a-zA-Z]", "", token)
    return token.isdigit()

def remove_unnecessary_tokens(token):
  token = re.sub(r"[^0-9a-zA-Z]","", token)
  return token.isdigit()

def remove_numerical(sentence):
  # create a regular expression pattern that matches any word or token that contains digits
  pattern = re.compile(r'\b\w*\d\w*\b')
  # replace all the matches with an empty string
  result = pattern.sub('', sentence)
  # return the result after stripping any extra spaces
  return result.strip()

def remove_alphanumerical(sentence):
  # create a regular expression pattern that matches any word or token that contains digits and is not a single digit
  pattern = re.compile(r'\b(?!\d\b)\w*\d\w*\b')
  # replace all the matches with an empty string
  result = pattern.sub('', sentence)
  # return the result after stripping any extra spaces
  return result.strip()
def remove_alphanumerical_2(text):
    result = text.split()
    result = [result for result in result if result.isalnum()]
    result = ' '.join(result,)
    return result

def remove_alphanumerical_3(text):
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    result = []
    for token in doc:
        if any(c.isalpha() or c.isnumeric() for c in token.text):
            result.append(token.text)
    result = ' '.join(result)
    return result
def is_number(filtered_sentence):

  # Reject numerical sentences
  if re.fullmatch(r"\d+(\.\d+)*", filtered_sentence):
    return False

  # Reject whitespace only
  if re.fullmatch(r"\s+", filtered_sentence):
    return False

  filtered_sentence = re.sub(r"\W", " ", filtered_sentence)
  filtered_sentence = re.sub(r"\s+", " ", filtered_sentence)

  return filtered_sentence.strip() == ''
# from rich import print as rprint
class TextWithAttention(nn.Module):
    def __init__(self, bert_model, num_classes, dropout_prob, hidden_size, num_heads):
        super(TextWithAttention, self).__init__()
        self.num_classes = num_classes
        self.bert = bert_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state

        sequence_output = sequence_output.transpose(0, 1)

        context, _ = self.attention(sequence_output, sequence_output, sequence_output)

        context = context.transpose(0, 1)

        context = self.dropout(context)

        context = context.mean(dim=1)

        logits = self.fc(context)

        return logits

def overall(ls):
    final = 2
    for lab in ls:
        if((lab == 0 and final == 2)):
            final = 0
        elif(lab == 1 and final != 1):
            final = 1
    return final


bert_model = BertModel.from_pretrained('bert-base-uncased')
model = TextWithAttention(bert_model=bert_model, num_classes=3, dropout_prob=0.5, hidden_size=768, num_heads=12)
model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.eval()
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
smodel = SentenceTransformer('all-MiniLM-L12-v2')
nlp = English()
nlp.add_pipe("sentencizer")
if device == 'cuda':
    print(f"device : {device}")
else:
    print(f"device : {device}")
# Rest of the functions...


class Worker(QObject):
    finished = pyqtSignal()
    def __init__(self, files, output_folder, input_spin_value,length):
        super().__init__()
        self.files = files
        self.output_folder = output_folder
        self.input_spin_value = input_spin_value
        self.length =length

    def run(self):
        print("worker started")
        print(self.length)

        self.file_progress = {}
        self.layouts = {}
        print(self.output_folder)
        QtCore.QCoreApplication.processEvents()


        for i, fname in enumerate(self.files):
            print(i+1)
            

            file_stem = fname
            pdf = pdfium.PdfDocument(f"{file_stem}")
            whole_text = ""
            n_pages = len(pdf)
            for num in range(n_pages):
                page = pdf[num]
                textpage = page.get_textpage()
                text_all = textpage.get_text_range()
                textpage = text_all
                textpage = clean_non_ascii_chars(textpage)
                textpage = replace_unicode_quotes(textpage)
                textpage = clean(textpage,bullets = True)
                textpage = clean(textpage,dashes=True)
                textpage = clean(textpage,extra_whitespace=True)
                textpage = clean(textpage,lowercase=True)
                textpage = group_broken_paragraphs(textpage)
                textpage = textpage.encode("utf-8")
                textpage = textpage.decode("utf-8")

                whole_text += textpage
            count = 0
            ls = []

            doc = nlp(whole_text)
            for sen in doc.sents:
                sen = str(sen)
                sen = remove_punctuation(sen)
                sen = sen.strip()
                if not remove_unnecessary_tokens(sen):
                    filtered_sentence = ""
                    filtered_sentence = sen
                    filtered_sentence = filtered_sentence.strip()
                    filtered_sentence = re.sub(r"[^0-9a-zA-Z!@#$%^&*()\-=_+`~\[\]{}\|'\";:,.<>?/\\]", " ", filtered_sentence)
                    # filtered_sentence = remove_alphanumerical_3(filtered_sentence)
                    filtered_sentence = re.sub(r"[^a-zA-Z]", " ", filtered_sentence)
                    filtered_sentence = clean(filtered_sentence,extra_whitespace=True)
                    count += 1
                    ls.append(filtered_sentence.strip())

            print(f"Number of candidates : {count}")
            print("\n")

            texts = ls

            tokenized_inputs = tokenizer(texts, truncation=True, padding=True,max_length=256)

            input_ids = torch.tensor(tokenized_inputs['input_ids'])
            attention_mask = torch.tensor(tokenized_inputs['attention_mask'])

            test_dataset = TensorDataset(input_ids, attention_mask)
            batch_size = int(self.input_spin_value)
            print(batch_size)
            test_dataloader = DataLoader(test_dataset, batch_size= batch_size , shuffle=False)

            predictions = []
            upd = 0
            with torch.no_grad():
                with tqdm(total=len(test_dataset), desc="Processing batches") as pbar:
                    for batch in test_dataloader:
                        input_ids, attention_mask = batch
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        logits = model(input_ids, attention_mask=attention_mask)
                        probabilities = torch.softmax(logits, dim=1)
                        predicted_labels = torch.argmax(probabilities, dim=1)
                        predictions.extend(predicted_labels)
                        upd += len(predicted_labels)
                        pbar.update(len(predicted_labels))
            predicted_labels_list = []
            for label in predictions:
                predicted_labels_list.append(label)
            label = predicted_labels_list

            similarities = []
            ls1 = np.array(ls)
            embeddings = np.array(smodel.encode(ls1))
            for z in range(1, len(ls1)):
                    similarity = util.cos_sim(embeddings[z-1], embeddings[z])[0][0]
                    similarities.append(similarity)

            similarities = np.array(similarities)
            minimas = argrelextrema(similarities, np.less, order=1)[0]

            text_list = []
            count = 0
            c = 0
            sentances = []
            sentances_labels = []
            final_labels = []
            split_points = [each for each in minimas]
            for num, sentence in enumerate(ls1):
                if num + 1 in split_points:
                    if(text_list):
                        sentances.extend(text_list)
                        for i in range(c):
                            final_labels.append(overall(sentances_labels))
                    text_list = []
                    sentances_labels = []
                    sentances_labels.append(label[count])
                    text_list.append(sentence)
                    count += 1
                    c = 1
                else:
                    text_list.append(sentence)
                    sentances_labels.append(label[count])
                    count += 1
                    c += 1

            for x in range(c):
                final_labels.append(overall(sentances_labels))
            sentances.extend(text_list)

            conv = {1: "functional requirement", 0: "non-functional requirement", 2: "information"}
            final_labels = [conv[lab] for lab in final_labels]
            df = pd.DataFrame({'text': sentances, 'label': final_labels})
            file_stem = Path(fname).stem
            output_dir = self.output_folder
            print(output_dir)
            output_dir = output_dir + '/' + file_stem + '_final.csv'
            print(output_dir)
            df.to_csv(f"{output_dir}", index=False)
            QtCore.QCoreApplication.processEvents()
        # Emit finished signal
        self.finished.emit()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('REX')

        # Create widgets
        self.file_button = QPushButton('Select Files')
        self.folder_button = QPushButton('Select Output Folder')
        self.progress = QProgressBar()
        self.run_button = QPushButton('Run')
        self.input_spin = QSpinBox()
        self.input_spin.setMinimum(1)
        self.files = ""
        self.output_folder = ""
        self.input_value = ""
        self.F_length = 0
        # Create worker instance
        self.worker = Worker(self.files[:], self.output_folder, self.input_value,self.F_length)
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.file_button)
        layout.addWidget(self.folder_button)
        layout.addWidget(self.input_spin)
        layout.addWidget(self.run_button)

        # Central widget
        container = QWidget()
        container.setLayout(layout)
        self.main_layout = QVBoxLayout()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)
        self.file_button.clicked.connect(self.select_files)
        self.folder_button.clicked.connect(self.select_folder)
        self.input_value = self.input_spin.value()

        # self.run_button.clicked.connect(self.run)
        self.run_button.clicked.connect(self.run_long_task)
        # Connect worker signals
        self.input_spin.valueChanged.connect(self.update_input)

    def update_input(self, value):
        self.input_value = value

    def reset_progress(self):
        return 


    def select_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Select Files', filter="PDF Files (*.pdf)")
        self.worker.files = filenames  # Update the worker files attribute
        self.files = filenames
        self.F_length = len(self.files)

    def select_folder(self):
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.worker.output_folder = foldername  # Update the worker output_folder attribute
        self.output_folder = foldername

    def run_long_task(self):

        # Create new thread and worker
        self.thread = QThread()
        self.worker = Worker(self.files[:], self.output_folder, self.input_value,self.F_length)
        self.worker.moveToThread(self.thread)
        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        # self.thread.started.connect(self.set_progress)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start the thread
        self.thread.start()

        # Final resets
        self.run_button.setEnabled(False)
        self.thread.finished.connect(lambda: self.run_button.setEnabled(True))

    def run(self):

        self.file_button.clicked.connect(self.select_files)
        self.folder_button.clicked.connect(self.select_folder)
        self.run_button.clicked.connect(self.run_long_task)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())

