"""
CorpusCrafter: Gelişmiş NLP Entegrasyonu Modülü
-----------------------------------------
Bu modül, PDF'ten CSV veri seti oluşturucu için gelişmiş doğal dil işleme işlevselliği sağlar.
Transformer tabanlı modeller, çoklu dil desteği ve gelişmiş metin işleme tekniklerini destekler.
"""

import os
import logging
import unicodedata
import re
import json
import time
import random
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path

# Mevcut modülü içe aktar
import sys
sys.path.append('/home/ubuntu')
from fixed_pdf_to_csv_core import (
    TextProcessor, 
    logger, 
    SUPPORTED_LANGUAGES
)

# Bağımlılık kontrolü
def check_dependencies():
    """Gerekli bağımlılıkları kontrol et ve durumlarını döndür."""
    dependencies = {
        "spacy": False,
        "transformers": False,
        "stanza": False,
        "flair": False,
        "nltk": False,
        "langdetect": False,
        "yake": False,
        "torch": False
    }
    
    # spaCy
    try:
        import spacy
        dependencies["spacy"] = True
    except ImportError:
        pass
    
    # Transformers
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        pass
    
    # Stanza
    try:
        import stanza
        dependencies["stanza"] = True
    except ImportError:
        pass
    
    # Flair
    try:
        import flair
        dependencies["flair"] = True
    except ImportError:
        pass
    
    # NLTK
    try:
        import nltk
        dependencies["nltk"] = True
    except ImportError:
        pass
    
    # Langdetect
    try:
        import langdetect
        dependencies["langdetect"] = True
    except ImportError:
        pass
    
    # YAKE
    try:
        import yake
        dependencies["yake"] = True
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    return dependencies


class AdvancedNLPProcessor(TextProcessor):
    """Gelişmiş NLP işleme için genişletilmiş sınıf."""
    
    def __init__(self, 
                 language: str = "en",
                 model_type: str = "transformer",
                 model_name: Optional[str] = None,
                 use_gpu: bool = False,
                 enable_ner: bool = True,
                 enable_sentiment: bool = True,
                 enable_summarization: bool = True,
                 enable_keyword_extraction: bool = True,
                 custom_pipeline: Optional[List[str]] = None,
                 cache_dir: Optional[str] = None):
        """
        Gelişmiş NLP işleyiciyi başlat.
        
        Args:
            language: ISO dil kodu (örn. 'en', 'tr')
            model_type: NLP model türü ('spacy', 'transformer', 'stanza', 'flair')
            model_name: Kullanılacak özel model adı (None ise dile göre varsayılan model)
            use_gpu: GPU kullanımını etkinleştir
            enable_ner: Varlık ismi tanıma işlemini etkinleştir
            enable_sentiment: Duygu analizi işlemini etkinleştir
            enable_summarization: Metin özetleme işlemini etkinleştir
            enable_keyword_extraction: Anahtar kelime çıkarma işlemini etkinleştir
            custom_pipeline: Özel NLP işlem hattı bileşenleri
            cache_dir: Model önbelleği için dizin
        """
        super().__init__(language)
        self.model_type = model_type
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.enable_ner = enable_ner
        self.enable_sentiment = enable_sentiment
        self.enable_summarization = enable_summarization
        self.enable_keyword_extraction = enable_keyword_extraction
        self.custom_pipeline = custom_pipeline
        self.cache_dir = cache_dir
        
        # Bağımlılıkları kontrol et
        self.dependencies = check_dependencies()
        
        # NLP modellerini yükle
        self.models = {}
        self._load_nlp_models()
        
        # Dil işleme araçlarını başlat
        self._initialize_language_tools()
        
    def _load_nlp_models(self):
        """Seçilen model türüne göre NLP modellerini yükle."""
        if self.model_type == "spacy":
            if not self.dependencies["spacy"]:
                logger.warning("spaCy yüklü değil. pip install spacy komutuyla yükleyin.")
                return
            self._load_spacy_models()
        elif self.model_type == "transformer":
            if not self.dependencies["transformers"]:
                logger.warning("transformers yüklü değil. pip install transformers komutuyla yükleyin.")
                return
            self._load_transformer_models()
        elif self.model_type == "stanza":
            if not self.dependencies["stanza"]:
                logger.warning("stanza yüklü değil. pip install stanza komutuyla yükleyin.")
                return
            self._load_stanza_models()
        elif self.model_type == "flair":
            if not self.dependencies["flair"]:
                logger.warning("flair yüklü değil. pip install flair komutuyla yükleyin.")
                return
            self._load_flair_models()
        else:
            logger.warning(f"Bilinmeyen model türü: {self.model_type}. Varsayılan olarak transformer kullanılıyor.")
            self.model_type = "transformer"
            if self.dependencies["transformers"]:
                self._load_transformer_models()
            else:
                logger.warning("transformers yüklü değil. pip install transformers komutuyla yükleyin.")
    
    def _load_spacy_models(self):
        """spaCy modellerini yükle."""
        try:
            import spacy
            from spacy.cli import download
            
            # Dile göre model seç
            if self.model_name:
                model_name = self.model_name
            else:
                model_map = {
                    "en": "en_core_web_trf" if self.use_gpu else "en_core_web_lg",
                    "tr": "tr_core_news_lg",
                    "de": "de_core_news_lg",
                    "fr": "fr_core_news_lg",
                    "es": "es_core_news_lg",
                    "it": "it_core_news_lg",
                    "nl": "nl_core_news_lg",
                    "pt": "pt_core_news_lg",
                    "ja": "ja_core_news_lg",
                    "zh": "zh_core_web_lg",
                    "ru": "ru_core_news_lg"
                }
                
                if self.language in model_map:
                    model_name = model_map[self.language]
                else:
                    logger.warning(f"Dil {self.language} için spaCy modeli bulunamadı. İngilizce model kullanılıyor.")
                    model_name = model_map["en"]
            
            # Model yüklü değilse indir
            try:
                self.models["main"] = spacy.load(model_name)
            except OSError:
                logger.info(f"spaCy modeli {model_name} indiriliyor...")
                download(model_name)
                self.models["main"] = spacy.load(model_name)
            
            # GPU kullanımını yapılandır
            if self.use_gpu and spacy.prefer_gpu():
                spacy.require_gpu()
                logger.info("spaCy GPU kullanımı etkinleştirildi")
            
            # Özel işlem hattı bileşenleri
            if self.custom_pipeline:
                # Mevcut işlem hattını devre dışı bırak
                disabled_pipes = [pipe for pipe in self.models["main"].pipe_names if pipe not in self.custom_pipeline]
                self.models["main"].disable_pipes(*disabled_pipes)
                logger.info(f"Özel spaCy işlem hattı etkinleştirildi: {', '.join(self.custom_pipeline)}")
            
            # Ek modeller
            if self.enable_sentiment and "sentiment" not in self.models["main"].pipe_names:
                # spaCy sentiment analizi için ek model
                try:
                    from spacytextblob.spacytextblob import SpacyTextBlob
                    self.models["main"].add_pipe("spacytextblob")
                    logger.info("spaCy duygu analizi bileşeni eklendi")
                except ImportError:
                    logger.warning("spacytextblob yüklü değil. Duygu analizi devre dışı.")
            
            if self.enable_summarization and self.dependencies["transformers"]:
                # Özetleme için transformers modeli
                try:
                    from transformers import pipeline
                    self.models["summarizer"] = pipeline(
                        "summarization", 
                        model="facebook/bart-large-cnn",
                        device=0 if self.use_gpu else -1
                    )
                    logger.info("Transformers özetleme modeli yüklendi")
                except Exception as e:
                    logger.warning(f"Özetleme modeli yüklenemedi: {str(e)}")
            
            logger.info(f"spaCy modeli {model_name} başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"spaCy modeli yüklenirken hata: {str(e)}")
    
    def _load_transformer_models(self):
        """Transformer modellerini yükle."""
        try:
            from transformers import (
                AutoTokenizer, 
                AutoModelForTokenClassification,
                AutoModelForSequenceClassification,
                AutoModelForSeq2SeqLM,
                pipeline
            )
            
            # Dile göre model seç
            lang_code = self.language
            
            # Ana model (token sınıflandırma - NER için)
            if self.enable_ner:
                if self.model_name:
                    ner_model_name = self.model_name
                else:
                    # Dile göre NER modeli seç
                    ner_model_map = {
                        "en": "dslim/bert-base-NER",
                        "tr": "savasy/bert-base-turkish-ner",
                        "de": "dbmdz/bert-base-german-cased-conll03-germeval",
                        "fr": "Jean-Baptiste/camembert-ner",
                        "es": "mrm8488/bert-spanish-cased-finetuned-ner",
                        "it": "Davlan/bert-base-multilingual-cased-ner-hrl",
                        "nl": "wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner",
                        "pt": "Davlan/bert-base-multilingual-cased-ner-hrl",
                        "ja": "cl-tohoku/bert-base-japanese-v2",
                        "zh": "ckiplab/bert-base-chinese-ner",
                        "ru": "DeepPavlov/bert-base-multilingual-cased-ner"
                    }
                    
                    if lang_code in ner_model_map:
                        ner_model_name = ner_model_map[lang_code]
                    else:
                        logger.warning(f"Dil {lang_code} için NER modeli bulunamadı. Çok dilli model kullanılıyor.")
                        ner_model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
                
                try:
                    self.models["ner"] = pipeline(
                        "ner", 
                        model=ner_model_name,
                        tokenizer=ner_model_name,
                        aggregation_strategy="simple",
                        device=0 if self.use_gpu else -1,
                        cache_dir=self.cache_dir
                    )
                    logger.info(f"NER modeli {ner_model_name} başarıyla yüklendi")
                except Exception as e:
                    logger.error(f"NER modeli yüklenirken hata: {str(e)}")
            
            # Duygu analizi modeli
            if self.enable_sentiment:
                if self.model_name and "sentiment" in self.model_name:
                    sentiment_model_name = self.model_name
                else:
                    # Dile göre duygu analizi modeli seç
                    sentiment_model_map = {
                        "en": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                        "tr": "savasy/bert-base-turkish-sentiment",
                        "de": "oliverguhr/german-sentiment-bert",
                        "fr": "tblard/tf-allocine",
                        "es": "pysentimiento/robertuito-sentiment-analysis",
                        "it": "neuraly/bert-base-italian-cased-sentiment",
                        "nl": "wietsedv/bert-base-dutch-cased-finetuned-sentiment",
                        "pt": "pysentimiento/robertuito-sentiment-analysis",
                        "ja": "cl-tohoku/bert-base-japanese-v2",
                        "zh": "uer/roberta-base-finetuned-jd-binary-chinese",
                        "ru": "blanchefort/rubert-base-cased-sentiment"
                    }
                    
                    if lang_code in sentiment_model_map:
                        sentiment_model_name = sentiment_model_map[lang_code]
                    else:
                        logger.warning(f"Dil {lang_code} için duygu analizi modeli bulunamadı. İngilizce model kullanılıyor.")
                        sentiment_model_name = sentiment_model_map["en"]
                
                try:
                    self.models["sentiment"] = pipeline(
                        "sentiment-analysis", 
                        model=sentiment_model_name,
                        tokenizer=sentiment_model_name,
                        device=0 if self.use_gpu else -1,
                        cache_dir=self.cache_dir
                    )
                    logger.info(f"Duygu analizi modeli {sentiment_model_name} başarıyla yüklendi")
                except Exception as e:
                    logger.error(f"Duygu analizi modeli yüklenirken hata: {str(e)}")
            
            # Özetleme modeli
            if self.enable_summarization:
                if self.model_name and "summarization" in self.model_name:
                    summarization_model_name = self.model_name
                else:
                    # Dile göre özetleme modeli seç
                    summarization_model_map = {
                        "en": "facebook/bart-large-cnn",
                        "tr": "ozcangundes/mt5-small-turkish-summarization",
                        "de": "T-Systems-onsite/mt5-small-sum-de-en",
                        "fr": "plguillou/t5-base-fr-sum-cnndm",
                        "es": "mrm8488/bert2bert_shared-spanish-finetuned-summarization",
                        "it": "Narrativa/bsc-sum-it",
                        "nl": "ml6team/mt5-small-dutch-summarization",
                        "pt": "unicamp-dl/mt5-small-portuguese-summarization",
                        "ja": "ku-nlp/bart-base-japanese",
                        "zh": "uer/t5-base-chinese-cluecorpussmall",
                        "ru": "IlyaGusev/mbart_ru_sum_gazeta"
                    }
                    
                    if lang_code in summarization_model_map:
                        summarization_model_name = summarization_model_map[lang_code]
                    else:
                        logger.warning(f"Dil {lang_code} için özetleme modeli bulunamadı. İngilizce model kullanılıyor.")
                        summarization_model_name = summarization_model_map["en"]
                
                try:
                    self.models["summarizer"] = pipeline(
                        "summarization", 
                        model=summarization_model_name,
                        device=0 if self.use_gpu else -1,
                        cache_dir=self.cache_dir
                    )
                    logger.info(f"Özetleme modeli {summarization_model_name} başarıyla yüklendi")
                except Exception as e:
                    logger.error(f"Özetleme modeli yüklenirken hata: {str(e)}")
            
            # Anahtar kelime çıkarma için model
            if self.enable_keyword_extraction and self.dependencies["yake"]:
                try:
                    import yake
                    self.models["keyword_extractor"] = yake.KeywordExtractor(
                        lan=lang_code if lang_code in yake.supported_languages else "en",
                        n=3,  # n-gram boyutu
                        dedupLim=0.9,  # tekrarlanan kelimeleri filtreleme eşiği
                        dedupFunc='seqm',  # tekrarlanan kelimeleri filtreleme fonksiyonu
                        windowsSize=1,  # pencere boyutu
                        top=10,  # çıkarılacak anahtar kelime sayısı
                        features=None
                    )
                    logger.info("YAKE anahtar kelime çıkarıcı başarıyla yüklendi")
                except Exception as e:
                    logger.error(f"YAKE anahtar kelime çıkarıcı yüklenirken hata: {str(e)}")
            
            logger.info("Transformer modelleri başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"Transformer modelleri yüklenirken hata: {str(e)}")
    
    def _load_stanza_models(self):
        """Stanza modellerini yükle."""
        if not self.dependencies["stanza"]:
            return
            
        try:
            import stanza
            from stanza.pipeline.core import DownloadMethod
            
            # Dile göre model seç
            lang_code = self.language
            
            # Stanza'nın desteklediği dil kodlarına dönüştür
            stanza_lang_map = {
                "en": "en",
                "tr": "tr",
                "de": "de",
                "fr": "fr",
                "es": "es",
                "it": "it",
                "nl": "nl",
                "pt": "pt",
                "ja": "ja",
                "zh": "zh",
                "ru": "ru"
            }
            
            if lang_code in stanza_lang_map:
                stanza_lang = stanza_lang_map[lang_code]
            else:
                logger.warning(f"Dil {lang_code} Stanza tarafından desteklenmiyor. İngilizce kullanılıyor.")
                stanza_lang = "en"
            
            # İşlem hattı bileşenlerini yapılandır
            processors = "tokenize,pos,lemma"
            if self.enable_ner:
                processors += ",ner"
            if self.custom_pipeline:
                processors = ",".join(self.custom_pipeline)
            
            # Model yüklü değilse indir
            try:
                stanza.download(
                    stanza_lang, 
                    processors=processors,
                    download_method=DownloadMethod.REUSE_RESOURCES if self.cache_dir else DownloadMethod.DOWNLOAD_RESOURCES,
                    dir=self.cache_dir
                )
            except Exception as e:
                logger.warning(f"Stanza modeli indirme hatası: {str(e)}. Mevcut modeller kullanılacak.")
            
            # Stanza işlem hattını başlat
            self.models["main"] = stanza.Pipeline(
                lang=stanza_lang,
                processors=processors,
                use_gpu=self.use_gpu,
                dir=self.cache_dir
            )
            
            # Ek modeller
            if self.enable_sentiment and "sentiment" not in processors and self.dependencies["transformers"]:
                try:
                    from transformers import pipeline
                    self.models["sentiment"] = pipeline(
                        "sentiment-analysis", 
                        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                        device=0 if self.use_gpu else -1
                    )
                    logger.info("Transformers duygu analizi modeli yüklendi")
                except Exception as e:
                    logger.warning(f"Transformers duygu analizi modeli yüklenemedi: {str(e)}")
            
            if self.enable_summarization and self.dependencies["transformers"]:
                try:
                    from transformers import pipeline
                    self.models["summarizer"] = pipeline(
                        "summarization", 
                        model="facebook/bart-large-cnn",
                        device=0 if self.use_gpu else -1
                    )
                    logger.info("Transformers özetleme modeli yüklendi")
                except Exception as e:
                    logger.warning(f"Transformers özetleme modeli yüklenemedi: {str(e)}")
            
            logger.info(f"Stanza modeli {stanza_lang} başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"Stanza modeli yüklenirken hata: {str(e)}")
    
    def _load_flair_models(self):
        """Flair modellerini yükle."""
        if not self.dependencies["flair"] or not self.dependencies["torch"]:
            return
            
        try:
            import flair
            import torch
            from flair.models import SequenceTagger
            from flair.embeddings import TransformerDocumentEmbeddings
            from flair.data import Sentence
            
            # GPU kullanımını yapılandır
            if self.use_gpu:
                flair.device = torch.device('cuda:0')
                logger.info("Flair GPU kullanımı etkinleştirildi")
            else:
                flair.device = torch.device('cpu')
            
            # NER modeli
            if self.enable_ner:
                if self.model_name:
                    ner_model_name = self.model_name
                else:
                    # Dile göre NER modeli seç
                    ner_model_map = {
                        "en": "flair/ner-english-large",
                        "de": "flair/ner-german-large",
                        "fr": "flair/ner-french",
                        "nl": "flair/ner-dutch-large",
                        "es": "flair/ner-spanish-large",
                        "it": "flair/ner-italian",
                        "pt": "flair/ner-portuguese-large",
                        "tr": "flair/ner-multi-fast",  # Çok dilli model
                        "ja": "flair/ner-multi-fast",  # Çok dilli model
                        "zh": "flair/ner-multi-fast",  # Çok dilli model
                        "ru": "flair/ner-russian"
                    }
                    
                    if self.language in ner_model_map:
                        ner_model_name = ner_model_map[self.language]
                    else:
                        logger.warning(f"Dil {self.language} için Flair NER modeli bulunamadı. Çok dilli model kullanılıyor.")
                        ner_model_name = "flair/ner-multi-fast"
                
                try:
                    self.models["ner"] = SequenceTagger.load(ner_model_name)
                    logger.info(f"Flair NER modeli {ner_model_name} başarıyla yüklendi")
                except Exception as e:
                    logger.error(f"Flair NER modeli yüklenirken hata: {str(e)}")
            
            # Duygu analizi modeli
            if self.enable_sentiment:
                try:
                    self.models["sentiment"] = flair.models.TextClassifier.load('en-sentiment')
                    logger.info("Flair duygu analizi modeli başarıyla yüklendi")
                except Exception as e:
                    logger.error(f"Flair duygu analizi modeli yüklenirken hata: {str(e)}")
            
            # Belge gömme modeli
            try:
                self.models["embeddings"] = TransformerDocumentEmbeddings('distilbert-base-uncased')
                logger.info("Flair belge gömme modeli başarıyla yüklendi")
            except Exception as e:
                logger.error(f"Flair belge gömme modeli yüklenirken hata: {str(e)}")
            
            logger.info("Flair modelleri başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"Flair modelleri yüklenirken hata: {str(e)}")
    
    def _initialize_language_tools(self):
        """Dil işleme araçlarını başlat."""
        # Dil algılama
        self.has_langdetect = self.dependencies["langdetect"]
        if not self.has_langdetect:
            logger.warning("langdetect yüklü değil. Dil algılama sınırlı olacak.")
        
        # Dil normalizasyonu
        self.has_nltk = self.dependencies["nltk"]
        if self.has_nltk:
            try:
                import nltk
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                from nltk.corpus import stopwords
                from nltk.stem import SnowballStemmer, WordNetLemmatizer
                
                # Dile göre araçları yapılandır
                if self.language in ['en', 'tr', 'de', 'fr', 'es', 'it', 'nl', 'pt', 'ru']:
                    self.stopwords = set(stopwords.words(self._map_language_code_to_nltk(self.language)))
                else:
                    self.stopwords = set(stopwords.words('english'))
                    logger.warning(f"Dil {self.language} için durma kelimeleri bulunamadı. İngilizce kullanılıyor.")
                
                # Kök bulma
                if self.language in ['en', 'tr', 'de', 'fr', 'es', 'it', 'nl', 'pt', 'ru']:
                    self.stemmer = SnowballStemmer(self._map_language_code_to_nltk(self.language))
                else:
                    self.stemmer = None
                    logger.warning(f"Dil {self.language} için kök bulucu bulunamadı.")
                
                # Lemmatizer (şu anda sadece İngilizce için)
                if self.language == 'en':
                    self.lemmatizer = WordNetLemmatizer()
                else:
                    self.lemmatizer = None
                
            except Exception as e:
                logger.warning(f"NLTK araçları başlatılırken hata: {str(e)}")
                self.stopwords = set()
                self.stemmer = None
                self.lemmatizer = None
        else:
            logger.warning("nltk yüklü değil. Metin normalizasyonu sınırlı olacak.")
            self.stopwords = set()
            self.stemmer = None
            self.lemmatizer = None
    
    def _map_language_code_to_nltk(self, lang_code):
        """ISO dil kodunu NLTK dil adına dönüştür."""
        nltk_lang_map = {
            "en": "english",
            "tr": "turkish",
            "de": "german",
            "fr": "french",
            "es": "spanish",
            "it": "italian",
            "nl": "dutch",
            "pt": "portuguese",
            "ru": "russian"
        }
        
        return nltk_lang_map.get(lang_code, "english")
    
    def preprocess_with_advanced_nlp(self, text: str) -> str:
        """
        Gelişmiş NLP kullanarak metin ön işleme.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            İşlenmiş metin
            
        Raises:
            ValueError: NLP modeli yüklü değilse
        """
        if not self.models or "main" not in self.models:
            logger.warning(f"{self.model_type} modeli yüklü değil, standart işleme kullanılıyor")
            return self.preprocess_text_advanced(text)
        
        logger.info(f"{self.model_type} tabanlı gelişmiş metin ön işleme başlatılıyor")
        
        # Model türüne göre işleme
        if self.model_type == "spacy":
            return self._process_with_spacy(text)
        elif self.model_type == "transformer":
            return self._process_with_transformer(text)
        elif self.model_type == "stanza":
            return self._process_with_stanza(text)
        elif self.model_type == "flair":
            return self._process_with_flair(text)
        else:
            logger.warning(f"Bilinmeyen model türü: {self.model_type}. Varsayılan işleme kullanılıyor.")
            return self.preprocess_text_advanced(text)
    
    def _process_with_spacy(self, text: str) -> str:
        """spaCy ile metin işle."""
        # Unicode normalizasyonu
        text = unicodedata.normalize('NFKC', text)
        
        # spaCy ile işle
        doc = self.models["main"](text)
        
        # Başlıkları ve soruları tanımla
        headings = []
        questions = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if self.is_heading(sent_text):
                headings.append(sent_text)
            elif self.is_question(sent_text):
                questions.append(sent_text)
        
        # Başlıkları ve soruları filtrele
        processed_paragraphs = []
        current_paragraph = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Başlıkları ve soruları atla
            if sent_text in headings or sent_text in questions:
                continue
            
            # Çok kısa cümleleri atla
            if len(sent_text) < 20:
                continue
            
            # Bu yeni bir paragrafın başlangıcıysa
            if sent.start_char > 0 and doc.text[sent.start_char-1] == '\n':
                if current_paragraph:
                    processed_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            # Durma kelimelerini filtrele (isteğe bağlı)
            # filtered_tokens = [token.text for token in sent if not token.is_stop]
            # sent_text = ' '.join(filtered_tokens)
            
            current_paragraph.append(sent_text)
        
        # Son paragrafı ekle
        if current_paragraph:
            processed_paragraphs.append(' '.join(current_paragraph))
        
        # Paragrafları birleştir
        processed_text = '\n\n'.join(processed_paragraphs)
        
        logger.info("spaCy tabanlı gelişmiş metin ön işleme tamamlandı")
        return processed_text
    
    def _process_with_transformer(self, text: str) -> str:
        """Transformer modelleri ile metin işle."""
        # Unicode normalizasyonu
        text = unicodedata.normalize('NFKC', text)
        
        # Metni cümlelere böl
        if self.has_nltk:
            import nltk
            try:
                sentences = nltk.sent_tokenize(text, language=self._map_language_code_to_nltk(self.language))
            except:
                sentences = nltk.sent_tokenize(text)
        else:
            # Basit cümle bölme
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Başlıkları ve soruları tanımla
        headings = []
        questions = []
        
        for sent in sentences:
            sent = sent.strip()
            if self.is_heading(sent):
                headings.append(sent)
            elif self.is_question(sent):
                questions.append(sent)
        
        # Başlıkları ve soruları filtrele
        processed_paragraphs = []
        current_paragraph = []
        
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            
            # Başlıkları ve soruları atla
            if sent in headings or sent in questions:
                continue
            
            # Çok kısa cümleleri atla
            if len(sent) < 20:
                continue
            
            # Bu yeni bir paragrafın başlangıcıysa
            if i > 0 and sentences[i-1].endswith(('.', '!', '?', ':', ';')):
                if current_paragraph:
                    processed_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            current_paragraph.append(sent)
        
        # Son paragrafı ekle
        if current_paragraph:
            processed_paragraphs.append(' '.join(current_paragraph))
        
        # Paragrafları birleştir
        processed_text = '\n\n'.join(processed_paragraphs)
        
        logger.info("Transformer tabanlı gelişmiş metin ön işleme tamamlandı")
        return processed_text
    
    def _process_with_stanza(self, text: str) -> str:
        """Stanza ile metin işle."""
        # Unicode normalizasyonu
        text = unicodedata.normalize('NFKC', text)
        
        # Stanza ile işle
        doc = self.models["main"](text)
        
        # Başlıkları ve soruları tanımla
        headings = []
        questions = []
        
        for sent in doc.sentences:
            sent_text = sent.text.strip()
            if self.is_heading(sent_text):
                headings.append(sent_text)
            elif self.is_question(sent_text):
                questions.append(sent_text)
        
        # Başlıkları ve soruları filtrele
        processed_paragraphs = []
        current_paragraph = []
        
        for sent in doc.sentences:
            sent_text = sent.text.strip()
            
            # Başlıkları ve soruları atla
            if sent_text in headings or sent_text in questions:
                continue
            
            # Çok kısa cümleleri atla
            if len(sent_text) < 20:
                continue
            
            # Yeni paragraf kontrolü (basit yaklaşım)
            if current_paragraph and sent_text.startswith(('The', 'A', 'In', 'On', 'At', 'Bu', 'Bir', 'Ve')):
                processed_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            
            current_paragraph.append(sent_text)
        
        # Son paragrafı ekle
        if current_paragraph:
            processed_paragraphs.append(' '.join(current_paragraph))
        
        # Paragrafları birleştir
        processed_text = '\n\n'.join(processed_paragraphs)
        
        logger.info("Stanza tabanlı gelişmiş metin ön işleme tamamlandı")
        return processed_text
    
    def _process_with_flair(self, text: str) -> str:
        """Flair ile metin işle."""
        # Unicode normalizasyonu
        text = unicodedata.normalize('NFKC', text)
        
        # Metni cümlelere böl
        if self.has_nltk:
            import nltk
            try:
                sentences = nltk.sent_tokenize(text, language=self._map_language_code_to_nltk(self.language))
            except:
                sentences = nltk.sent_tokenize(text)
        else:
            # Basit cümle bölme
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Flair cümleleri oluştur
        from flair.data import Sentence
        flair_sentences = [Sentence(sent) for sent in sentences]
        
        # Başlıkları ve soruları tanımla
        headings = []
        questions = []
        
        for sent in sentences:
            sent = sent.strip()
            if self.is_heading(sent):
                headings.append(sent)
            elif self.is_question(sent):
                questions.append(sent)
        
        # Başlıkları ve soruları filtrele
        processed_paragraphs = []
        current_paragraph = []
        
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            
            # Başlıkları ve soruları atla
            if sent in headings or sent in questions:
                continue
            
            # Çok kısa cümleleri atla
            if len(sent) < 20:
                continue
            
            # Bu yeni bir paragrafın başlangıcıysa
            if i > 0 and sentences[i-1].endswith(('.', '!', '?', ':', ';')):
                if current_paragraph:
                    processed_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            current_paragraph.append(sent)
        
        # Son paragrafı ekle
        if current_paragraph:
            processed_paragraphs.append(' '.join(current_paragraph))
        
        # Paragrafları birleştir
        processed_text = '\n\n'.join(processed_paragraphs)
        
        logger.info("Flair tabanlı gelişmiş metin ön işleme tamamlandı")
        return processed_text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Metinden varlık isimlerini çıkar.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            Varlık listesi (tür, metin, konum)
        """
        if not self.enable_ner or "ner" not in self.models:
            logger.warning("NER etkin değil veya model yüklü değil")
            return []
        
        logger.info("Varlık ismi tanıma başlatılıyor")
        
        entities = []
        
        try:
            if self.model_type == "spacy":
                doc = self.models["main"](text)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "type": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
            
            elif self.model_type == "transformer":
                ner_results = self.models["ner"](text)
                for entity in ner_results:
                    entities.append({
                        "text": entity["word"],
                        "type": entity["entity"],
                        "score": entity["score"],
                        "start": entity["start"],
                        "end": entity["end"]
                    })
            
            elif self.model_type == "stanza":
                doc = self.models["main"](text)
                for sent in doc.sentences:
                    for ent in sent.ents:
                        entities.append({
                            "text": ent.text,
                            "type": ent.type,
                            "start": -1,  # Stanza doğrudan karakter konumlarını sağlamaz
                            "end": -1
                        })
            
            elif self.model_type == "flair":
                from flair.data import Sentence
                sentence = Sentence(text)
                self.models["ner"].predict(sentence)
                for entity in sentence.get_spans('ner'):
                    entities.append({
                        "text": entity.text,
                        "type": entity.tag,
                        "score": entity.score,
                        "start": -1,  # Flair doğrudan karakter konumlarını sağlamaz
                        "end": -1
                    })
        
        except Exception as e:
            logger.error(f"Varlık ismi tanıma hatası: {str(e)}")
        
        logger.info(f"Varlık ismi tanıma tamamlandı. {len(entities)} varlık bulundu.")
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Metnin duygu analizini yap.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            Duygu analizi sonucu (polarite, nesnellik)
        """
        if not self.enable_sentiment or "sentiment" not in self.models:
            logger.warning("Duygu analizi etkin değil veya model yüklü değil")
            return {"polarity": 0.0, "subjectivity": 0.0}
        
        logger.info("Duygu analizi başlatılıyor")
        
        result = {"polarity": 0.0, "subjectivity": 0.0, "label": "NEUTRAL"}
        
        try:
            if self.model_type == "spacy":
                doc = self.models["main"](text)
                if hasattr(doc, "_.blob"):
                    result["polarity"] = doc._.blob.polarity
                    result["subjectivity"] = doc._.blob.subjectivity
                    if result["polarity"] > 0.1:
                        result["label"] = "POSITIVE"
                    elif result["polarity"] < -0.1:
                        result["label"] = "NEGATIVE"
            
            elif self.model_type == "transformer":
                sentiment_result = self.models["sentiment"](text)
                if sentiment_result:
                    result["label"] = sentiment_result[0]["label"]
                    result["score"] = sentiment_result[0]["score"]
                    
                    # Polarite değerini hesapla
                    if result["label"] == "POSITIVE":
                        result["polarity"] = result["score"]
                    elif result["label"] == "NEGATIVE":
                        result["polarity"] = -result["score"]
                    else:
                        result["polarity"] = 0.0
            
            elif self.model_type == "stanza":
                # Stanza doğrudan duygu analizi sağlamaz, transformers kullanılır
                if "sentiment" in self.models:
                    sentiment_result = self.models["sentiment"](text)
                    if sentiment_result:
                        result["label"] = sentiment_result[0]["label"]
                        result["score"] = sentiment_result[0]["score"]
                        
                        # Polarite değerini hesapla
                        if result["label"] == "POSITIVE":
                            result["polarity"] = result["score"]
                        elif result["label"] == "NEGATIVE":
                            result["polarity"] = -result["score"]
                        else:
                            result["polarity"] = 0.0
            
            elif self.model_type == "flair":
                from flair.data import Sentence
                sentence = Sentence(text)
                self.models["sentiment"].predict(sentence)
                result["label"] = sentence.labels[0].value
                result["score"] = sentence.labels[0].score
                
                # Polarite değerini hesapla
                if result["label"] == "POSITIVE":
                    result["polarity"] = result["score"]
                elif result["label"] == "NEGATIVE":
                    result["polarity"] = -result["score"]
                else:
                    result["polarity"] = 0.0
        
        except Exception as e:
            logger.error(f"Duygu analizi hatası: {str(e)}")
        
        logger.info(f"Duygu analizi tamamlandı. Sonuç: {result['label']}")
        return result
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Metni özetle.
        
        Args:
            text: Özetlenecek metin
            max_length: Maksimum özet uzunluğu
            min_length: Minimum özet uzunluğu
            
        Returns:
            Özetlenmiş metin
        """
        if not self.enable_summarization or "summarizer" not in self.models:
            logger.warning("Özetleme etkin değil veya model yüklü değil")
            return text[:max_length] + "..."
        
        logger.info("Metin özetleme başlatılıyor")
        
        try:
            # Metin çok kısaysa özetleme yapma
            if len(text.split()) < min_length:
                logger.info("Metin çok kısa, özetleme yapılmadı")
                return text
            
            summary = self.models["summarizer"](
                text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False
            )
            
            if summary and len(summary) > 0:
                logger.info("Metin özetleme tamamlandı")
                return summary[0]["summary_text"]
            else:
                logger.warning("Özetleme sonuç vermedi, orijinal metin döndürülüyor")
                return text
        
        except Exception as e:
            logger.error(f"Metin özetleme hatası: {str(e)}")
            return text
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Metinden anahtar kelimeleri çıkar.
        
        Args:
            text: İşlenecek metin
            top_n: Çıkarılacak anahtar kelime sayısı
            
        Returns:
            Anahtar kelime ve puan çiftleri listesi
        """
        if not self.enable_keyword_extraction or "keyword_extractor" not in self.models:
            logger.warning("Anahtar kelime çıkarma etkin değil veya model yüklü değil")
            return []
        
        logger.info("Anahtar kelime çıkarma başlatılıyor")
        
        try:
            keywords = self.models["keyword_extractor"].extract_keywords(text)
            
            # En iyi N anahtar kelimeyi al
            top_keywords = keywords[:top_n]
            
            logger.info(f"Anahtar kelime çıkarma tamamlandı. {len(top_keywords)} anahtar kelime bulundu.")
            return top_keywords
        
        except Exception as e:
            logger.error(f"Anahtar kelime çıkarma hatası: {str(e)}")
            return []
    
    def normalize_text(self, text: str, remove_stopwords: bool = False, 
                      stemming: bool = False, lemmatization: bool = False) -> str:
        """
        Metni normalize et.
        
        Args:
            text: İşlenecek metin
            remove_stopwords: Durma kelimelerini kaldır
            stemming: Kök bulma uygula
            lemmatization: Lemmatization uygula
            
        Returns:
            Normalize edilmiş metin
        """
        if not self.has_nltk:
            logger.warning("NLTK yüklü değil, sınırlı normalizasyon uygulanıyor")
            return text
        
        logger.info("Metin normalizasyonu başlatılıyor")
        
        import nltk
        from nltk.tokenize import word_tokenize
        
        try:
            # Metni kelimelere böl
            words = word_tokenize(text, language=self._map_language_code_to_nltk(self.language))
            
            # Durma kelimelerini kaldır
            if remove_stopwords and self.stopwords:
                words = [word for word in words if word.lower() not in self.stopwords]
            
            # Kök bulma
            if stemming and self.stemmer:
                words = [self.stemmer.stem(word) for word in words]
            
            # Lemmatization
            if lemmatization and self.lemmatizer and self.language == 'en':
                words = [self.lemmatizer.lemmatize(word) for word in words]
            
            # Kelimeleri birleştir
            normalized_text = ' '.join(words)
            
            logger.info("Metin normalizasyonu tamamlandı")
            return normalized_text
        
        except Exception as e:
            logger.error(f"Metin normalizasyonu hatası: {str(e)}")
            return text
    
    def detect_language(self, text: str) -> str:
        """
        Metnin dilini algıla.
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            ISO dil kodu (örn. 'en', 'tr')
        """
        if not self.has_langdetect:
            logger.warning("langdetect yüklü değil, dil algılama yapılamıyor")
            return self.language
        
        try:
            import langdetect
            
            # Daha hızlı algılama için sadece ilk 1000 karakteri kullan
            sample = text[:1000]
            lang = langdetect.detect(sample)
            logger.info(f"Algılanan dil: {lang}")
            
            return lang
        
        except Exception as e:
            logger.warning(f"Dil algılama başarısız: {str(e)}. Varsayılan dil kullanılıyor.")
            return self.language
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Metnin yapısal analizini yap.
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            Yapısal analiz sonuçları
        """
        logger.info("Metin yapısı analizi başlatılıyor")
        
        import re
        
        # Cümlelere böl
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Paragrafları bul
        paragraphs = text.split('\n\n')
        
        # Başlıkları bul
        headings = []
        for line in text.split('\n'):
            if self.is_heading(line):
                headings.append(line.strip())
        
        # Soruları bul
        questions = []
        for sentence in sentences:
            if self.is_question(sentence):
                questions.append(sentence.strip())
        
        # Kelime sayısı
        words = text.split()
        word_count = len(words)
        
        # Ortalama cümle uzunluğu
        avg_sentence_length = word_count / max(1, len(sentences))
        
        # Ortalama kelime uzunluğu
        total_char_count = sum(len(word) for word in words)
        avg_word_length = total_char_count / max(1, word_count)
        
        # Sonuçları topla
        results = {
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "heading_count": len(headings),
            "question_count": len(questions),
            "word_count": word_count,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "headings": headings,
            "questions": questions
        }
        
        logger.info("Metin yapısı analizi tamamlandı")
        return results


# Test kodu
if __name__ == "__main__":
    # Bağımlılıkları kontrol et
    dependencies = check_dependencies()
    print("Bağımlılık durumu:")
    for dep, status in dependencies.items():
        print(f"  {dep}: {'Yüklü' if status else 'Yüklü değil'}")
    
    # Test için en az bir NLP kütüphanesi gerekli
    if not any([dependencies["spacy"], dependencies["transformers"], 
                dependencies["stanza"], dependencies["flair"]]):
        print("Hiçbir NLP kütüphanesi yüklü değil. En az birini yükleyin:")
        print("  pip install spacy")
        print("  pip install transformers")
        print("  pip install stanza")
        print("  pip install flair")
        sys.exit(1)
    
    # Gelişmiş NLP işleyiciyi test et
    model_type = "transformer" if dependencies["transformers"] else "spacy"
    advanced_processor = AdvancedNLPProcessor(
        language="en",
        model_type=model_type,
        use_gpu=False,
        enable_ner=True,
        enable_sentiment=True,
        enable_summarization=True,
        enable_keyword_extraction=True
    )
    
    # Test metni
    test_text = """
    Artificial Intelligence (AI) refers to systems that mimic human intelligence and can iteratively 
    improve themselves based on the data they collect. AI encompasses various subfields including 
    machine learning, deep learning, natural language processing, computer vision, and robotics.
    
    AI technologies are revolutionizing many sectors including healthcare, finance, education, 
    transportation, and manufacturing. For example, in healthcare, AI algorithms are being used 
    to diagnose diseases, recommend treatment plans, and accelerate drug discovery.
    
    The rapid advancement of AI raises important ethical questions about privacy, bias, 
    accountability, and the future of work. Researchers and policymakers are working to 
    develop frameworks to ensure that AI is developed and deployed in ways that are 
    beneficial, fair, and safe for society.
    """
    
    # Gelişmiş metin ön işleme
    processed_text = advanced_processor.preprocess_with_advanced_nlp(test_text)
    print("\nİşlenmiş Metin:")
    print(processed_text)
    
    # Varlık ismi tanıma
    if advanced_processor.enable_ner and "ner" in advanced_processor.models:
        entities = advanced_processor.extract_entities(test_text)
        print("\nVarlıklar:")
        for entity in entities[:5]:  # İlk 5 varlığı göster
            print(f"  {entity['text']} ({entity['type']})")
    
    # Duygu analizi
    if advanced_processor.enable_sentiment and "sentiment" in advanced_processor.models:
        sentiment = advanced_processor.analyze_sentiment(test_text)
        print("\nDuygu Analizi:")
        print(f"  Etiket: {sentiment['label']}")
        print(f"  Polarite: {sentiment.get('polarity', 'N/A')}")
        if 'score' in sentiment:
            print(f"  Puan: {sentiment['score']}")
    
    # Metin özetleme
    if advanced_processor.enable_summarization and "summarizer" in advanced_processor.models:
        summary = advanced_processor.summarize_text(test_text, max_length=100, min_length=30)
        print("\nÖzet:")
        print(f"  {summary}")
    
    # Anahtar kelime çıkarma
    if advanced_processor.enable_keyword_extraction and "keyword_extractor" in advanced_processor.models:
        keywords = advanced_processor.extract_keywords(test_text, top_n=5)
        print("\nAnahtar Kelimeler:")
        for keyword, score in keywords:
            print(f"  {keyword} ({score:.4f})")
    
    # Metin yapısı analizi
    structure = advanced_processor.analyze_text_structure(test_text)
    print("\nMetin Yapısı:")
    print(f"  Cümle Sayısı: {structure['sentence_count']}")
    print(f"  Paragraf Sayısı: {structure['paragraph_count']}")
    print(f"  Kelime Sayısı: {structure['word_count']}")
    print(f"  Ortalama Cümle Uzunluğu: {structure['avg_sentence_length']:.2f} kelime")
