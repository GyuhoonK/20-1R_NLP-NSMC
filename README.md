# NLP-NSMC
2020-1R Natural Language Processoogle 

Environment : Google Colab, Runtime : TPU<br>
<br>
Requirement : KorBert(ETRI), src_tokenizer.py(ETRI), vocab.korean_morp.list(ETRI) (downloaded http://aiopen.etri.re.kr/aidata_download.php<br>
              ETRI API AccessKey (issued http://aiopen.etri.re.kr/)<br>


filepath setiing: 1) nlp_path : folder which contains 'KoBERT_usingTPU_binary.ipynb', KorBERT config file('bert_config.json'), src_tokenizer which contains 'src_tokenizer.py' and kaggle dataset<br>
                  2) ckpt_path : the path to ETRI's 'pytorch_model.bin'<br>
<br>                  
<br>              

*modify these two function to the codes below in src_tokenizer.py<br>*

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



