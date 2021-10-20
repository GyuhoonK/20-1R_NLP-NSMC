# NLP-NSMC
2020-1R Natural Language Processoogle 

## Environment
Google Colab, Runtime : TPU<br>
<br>

## Requirement

1) `pytorch_model.bin`(ETRI),` src_tokenizer.py`(ETRI), `vocab.korean_morp.list`(ETRI) (download here http://aiopen.etri.re.kr/aidata_download.php<br>

2) ETRI API AccessKey (get here http://aiopen.etri.re.kr/)<br>

3) Kaggle Dataset(https://www.kaggle.com/c/cose461k)
<br>

## Filepath setting

1) `nlp_path` : folder which contains '`KoBERT_usingTPU_binary.ipynb`', KorBERT config file('`bert_config.json`'), model(`pytorch_model.bin`), `src_tokenizer` folder which contains '`src_tokenizer.py`' and kaggle dataset<br>

2) `ckpt_path` : the path to ETRI's '`pytorch_model.bin`'<br>

3) Kaggle Dataset(`ko_data.csv`, `ko_sample.csv`) shoud be saved in your `nlp_path`
<br>                  
<br>              

## *modify these two functions to the codes below in `src_tokenizer.py`<br>*

    def load_vocab(vocab_file) : 
        ~
        with tf.io.gfile.GFile(vocab_file, "r") as reader:
        ~

***
    def convert_by_vocab(vocab, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
          if item in vocab.keys():
            output.append(vocab[item])
          else : output.append(vocab['[UNK]'])
        return output


## SourceCode Reference

1) BertForSequenceClassification <br>
https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForSequenceClassification

2) TPU Setting<br>
https://github.com/GoogleCloudPlatform/training-data-analyst/issues/678
