"""
CorpusCrafter: PDF to AI Dataset Generator
-----------------------------------------
Bu modül, PDF'ten CSV veri seti oluşturucu için ana işlevselliği sağlar.
Metin çıkarma, ön işleme, bölümleme ve soru oluşturma işlemlerini içerir.
"""

import os
import sys
import time
import logging
import unicodedata
import re
import uuid
import argparse
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import importlib.metadata

# Gerekli kütüphaneler
import pandas as pd
import PyPDF2
from dotenv import load_dotenv
import openai
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter
)
from langchain.schema import Document

# İsteğe bağlı bağımlılıkları kontrol et
try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Versiyon bilgisi
__version__ = "0.2.1"

# Ortam değişkenlerini yükle
load_dotenv()

# Günlük kaydını yapılandır
def configure_logging(log_file="pdf_to_csv.log", log_level=logging.INFO):
    """
    Günlük kaydını yapılandır.
    
    Args:
        log_file: Günlük dosyasının adı
        log_level: Günlük seviyesi
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Varsayılan günlük kaydını yapılandır
logger = configure_logging()

# OpenAI API anahtarını kontrol et
def check_openai_api_key():
    """
    OpenAI API anahtarını kontrol et ve istemciyi yapılandır.
    
    Returns:
        OpenAI istemcisi
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY ortam değişkeni bulunamadı!")
        print("Hata: OPENAI_API_KEY ortam değişkeni ayarlanmamış. Lütfen API anahtarınızı ayarlayın.")
        print("Örnek: export OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    return openai.OpenAI(api_key=api_key)

# Test modu için sahte OpenAI istemcisi
class MockOpenAI:
    """
    Test için sahte OpenAI istemcisi.
    """
    class ChatCompletion:
        class Create:
            def __init__(self, choices):
                self.choices = choices
        
        @staticmethod
        def create(**kwargs):
            mock_response = MockOpenAI.ChatCompletion.Create(
                choices=[
                    type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': "Bu bir test sorusudur?"
                        })
                    })
                ]
            )
            return mock_response

    chat = type('obj', (object,), {'completions': ChatCompletion})

# OpenAI istemcisini yapılandır
client = None
TEST_MODE = os.environ.get("CORPUSCRAFTER_TEST_MODE", "false").lower() == "true"

if TEST_MODE:
    logger.info("Test modu etkin, sahte OpenAI istemcisi kullanılıyor")
    client = MockOpenAI()
else:
    client = check_openai_api_key()

# Kullanılabilir OpenAI modelleri
AVAILABLE_MODELS = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "Genel amaçlı kullanım için hızlı ve ekonomik model.",
        "max_tokens": 4096,
        "default_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 150}
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "GPT-4o'nun daha küçük ve daha hızlı versiyonu.",
        "max_tokens": 8192,
        "default_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 150}
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Yüksek kaliteli yanıtlar için ideal, en gelişmiş çok modlu model.",
        "max_tokens": 8192,
        "default_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 150}
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "description": "GPT-4'ün daha hızlı ve daha ekonomik versiyonu.",
        "max_tokens": 4096,
        "default_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 150}
    }
}

# Varsayılan model
DEFAULT_MODEL = "gpt-4o-mini"

# Metin bölme algoritmaları
TEXT_SPLITTER_TYPES = {
    "recursive": {
        "name": "Recursive Character Splitter",
        "description": "Metni belirli ayırıcılara göre özyinelemeli olarak böler.",
        "class": RecursiveCharacterTextSplitter
    },
    "sentence_transformers": {
        "name": "Sentence Transformers Token Splitter",
        "description": "Metni cümle transformers modelini kullanarak tokenlara böler.",
        "class": SentenceTransformersTokenTextSplitter
    },
    "token": {
        "name": "Token Splitter",
        "description": "Metni tokenlara göre böler.",
        "class": TokenTextSplitter
    }
}

# Varsayılan metin bölücü
DEFAULT_TEXT_SPLITTER = "recursive"

# Desteklenen diller ve özellikleri
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "heading_patterns": [
            r'^[A-Z0-9]',       # Büyük harf veya rakamla başlayan
            r'^[IVX]+\.',       # Roma rakamlarıyla başlayan (I., II., vb.)
            r'^\d+\.\s',        # Numaralandırılmış başlık (1. Introduction, vb.)
            r'^[A-Za-z]+\s\d+', # "Chapter 1", "Section 2" vb.
            r'^[A-Z\s]+$'       # Tamamen büyük harflerden oluşan
        ],
        "question_starters": [
            'what ', 'why ', 'how ', 'which ', 'who ', 'whom ', 'whose ',
            'where ', 'when ', 'can ', 'could ', 'would ', 'should ', 'is ',
            'are ', 'do ', 'does ', 'did ', 'have ', 'has ', 'had '
        ]
    },
    "tr": {
        "name": "Turkish",
        "heading_patterns": [
            r'^[A-Z0-9]',       # Büyük harfle veya rakamla başlayan
            r'^[IVX]+\.',       # Roma rakamları ile başlayan (I., II., vb.)
            r'^\d+\.\s',        # Numaralandırılmış başlık (1. Giriş, vb.)
            r'^[A-Za-z]+\s\d+', # "Bölüm 1", "Kısım 2" gibi
            r'^[A-Z\s]+$',      # Tamamen büyük harflerden oluşan
            r'^[A-Za-z]+\s\d+:' # "Bölüm 1:" gibi
        ],
        "question_starters": [
            'ne ', 'neden ', 'niçin ', 'niye ', 'nasıl ', 'hangi ', 'kim ', 'kime ', 'kimi ',
            'nerede ', 'ne zaman ', 'kaç ', 'kaçta ', 'nereye ', 'nereden ', 'mi ', 'mı ',
            'mu ', 'mü '
        ]
    },
    # Ek diller buraya eklenebilir
}

# Varsayılan dil
DEFAULT_LANGUAGE = "auto"

class PDFProcessor:
    """PDF dosyalarını işleme ve metin çıkarma sınıfı."""
    
    def __init__(self, input_dir: str = "input", language: str = DEFAULT_LANGUAGE):
        """
        PDF işleyiciyi başlat.
        
        Args:
            input_dir: PDF dosyalarını içeren dizin
            language: Belge dili (otomatik algılama için "auto")
        """
        self.input_dir = input_dir
        self.language = language
        
    def get_pdf_path(self, specific_file: Optional[str] = None) -> str:
        """
        Giriş dizininden bir PDF dosyası seç.
        
        Args:
            specific_file: İşlenecek belirli bir dosya adı (isteğe bağlı)
            
        Returns:
            Seçilen PDF'nin tam yolu
            
        Raises:
            FileNotFoundError: Belirtilen dosya veya dizinde PDF bulunamazsa
        """
        if specific_file:
            pdf_path = os.path.join(self.input_dir, specific_file)
            if not os.path.exists(pdf_path):
                error_msg = f"Belirtilen dosya bulunamadı: {pdf_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            return pdf_path
        
        # Dizindeki ilk PDF dosyasını bul
        try:
            pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        except FileNotFoundError:
            error_msg = f"Giriş dizini bulunamadı: {self.input_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not pdf_files:
            error_msg = f"{self.input_dir} dizininde PDF dosyası bulunamadı!"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # İlk PDF dosyasını seç
        selected_pdf = pdf_files[0]
        pdf_path = os.path.join(self.input_dir, selected_pdf)
        logger.info(f"Seçilen PDF: {pdf_path}")
        
        return pdf_path
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        PDF dosyasından metin çıkar ve gereksiz kısımları filtrele.
        
        Args:
            pdf_path: PDF dosyasının tam yolu
            
        Returns:
            Tuple[str, Dict[str, Any]]: Çıkarılan metin ve istatistikler
            
        Raises:
            Exception: PDF okuma hatası durumunda
        """
        logger.info(f"PDF dosyası okunuyor: {pdf_path}")
        
        stats = {
            "total_pages": 0,
            "skipped_pages": 0,
            "empty_pages": 0,
            "image_pages": 0
        }
        
        extracted_text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                stats["total_pages"] = len(reader.pages)
                
                # İlk birkaç ve son birkaç sayfayı atla (kapak, önsöz, dizin vb.)
                start_page = min(1, stats["total_pages"] - 1)  # Kapak sayfasını atla
                end_page = max(start_page, stats["total_pages"] - 2)  # Son sayfaları atla
                
                for i in range(start_page, end_page):
                    page = reader.pages[i]
                    page_text = page.extract_text()
                    
                    # Boş sayfaları atla
                    if not page_text or page_text.isspace():
                        stats["empty_pages"] += 1
                        stats["skipped_pages"] += 1
                        continue
                    
                    # Çoğunlukla görüntü içeren sayfaları algıla ve atla
                    # (Çok az metin içeren sayfalar genellikle görüntü ağırlıklıdır)
                    if len(page_text.split()) < 20:
                        stats["image_pages"] += 1
                        stats["skipped_pages"] += 1
                        continue
                    
                    # Dipnotları ve görüntü başlıklarını filtrele
                    # (Basit yaklaşım: Kısa satırları ve rakamla başlayan satırları atla)
                    filtered_lines = []
                    for line in page_text.split('\n'):
                        # Dipnot ve referans filtreleme
                        if re.match(r'^\d+\s', line) and len(line) < 100:
                            continue
                        # Çok kısa satırları atla (muhtemelen başlık veya başlık)
                        if len(line.strip()) < 5:
                            continue
                        filtered_lines.append(line)
                    
                    extracted_text += '\n'.join(filtered_lines) + '\n\n'
        
        except Exception as e:
            logger.error(f"PDF okuma hatası: {str(e)}")
            raise
        
        logger.info(f"PDF metin çıkarma tamamlandı. Toplam {stats['total_pages']} sayfa, "
                    f"{stats['skipped_pages']} sayfa atlandı.")
        
        return extracted_text, stats
    
    def detect_language(self, text: str) -> str:
        """
        Metnin dilini algıla.
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            ISO dil kodu (örn. 'en', 'tr')
        """
        if self.language != "auto":
            return self.language
        
        if not LANGDETECT_AVAILABLE:
            logger.warning("langdetect yüklü değil. Varsayılan olarak İngilizce kullanılıyor.")
            return "en"
        
        try:
            # Daha hızlı algılama için sadece ilk 1000 karakteri kullan
            sample = text[:1000]
            lang = langdetect.detect(sample)
            logger.info(f"Algılanan dil: {lang}")
            
            # Desteklenen dillere eşle veya varsayılan olarak İngilizce kullan
            if lang in SUPPORTED_LANGUAGES:
                return lang
            else:
                logger.warning(f"Algılanan dil {lang} tam olarak desteklenmiyor. İngilizce kalıplar kullanılıyor.")
                return "en"
        except Exception as e:
            logger.warning(f"Dil algılama başarısız: {str(e)}. Varsayılan olarak İngilizce kullanılıyor.")
            return "en"


class TextProcessor:
    """Metin ön işleme ve temizleme sınıfı."""
    
    def __init__(self, language: str = "en"):
        """
        Metin işleyiciyi başlat.
        
        Args:
            language: ISO dil kodu (örn. 'en', 'tr')
        """
        self.language = language
        
        # Dile özgü kalıpları al
        if language in SUPPORTED_LANGUAGES:
            self.lang_config = SUPPORTED_LANGUAGES[language]
        else:
            logger.warning(f"Dil {language} desteklenmiyor. İngilizce kalıplar kullanılıyor.")
            self.language = "en"
            self.lang_config = SUPPORTED_LANGUAGES["en"]
        
        # spaCy'yi başlat (mevcutsa)
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                if language == "en":
                    self.nlp = spacy.load("en_core_web_sm")
                elif language == "tr":
                    self.nlp = spacy.load("tr_core_news_sm")
                # Gerektiğinde daha fazla dil ekle
            except OSError:
                logger.warning(f"spaCy modeli {language} için bulunamadı. Model indirme talimatları:")
                if language == "en":
                    logger.warning("python -m spacy download en_core_web_sm")
                elif language == "tr":
                    logger.warning("python -m spacy download tr_core_news_sm")
    
    def is_heading(self, line: str) -> bool:
        """
        Bir satırın başlık olup olmadığını kontrol et.
        
        Args:
            line: Kontrol edilecek satır
            
        Returns:
            bool: Satır bir başlıksa True, değilse False
        """
        line = line.strip()
        
        if not line:
            return False
        
        # Çok uzun satırlar başlık değildir
        if len(line) > 100:
            return False
        
        # Nokta veya soru işareti içeren satırlar genellikle başlık değildir
        if '.' in line[:-1] or '?' in line:
            return False
        
        # Dile özgü başlık kalıplarına göre kontrol et
        for pattern in self.lang_config["heading_patterns"]:
            if re.match(pattern, line):
                return True
        
        return False
    
    def is_question(self, line: str) -> bool:
        """
        Bir satırın soru olup olmadığını kontrol et.
        
        Args:
            line: Kontrol edilecek satır
            
        Returns:
            bool: Satır bir soruysa True, değilse False
        """
        line = line.strip()
        
        if not line:
            return False
        
        # Soru işaretiyle biten satırlar
        if line.endswith('?'):
            return True
        
        # Soru kelimeleriyle başlayan satırlar
        lower_line = line.lower()
        for starter in self.lang_config["question_starters"]:
            if lower_line.startswith(starter):
                return True
        
        return False
    
    def preprocess_text_advanced(self, text: str) -> str:
        """
        Gelişmiş metin ön işleme: Başlıkları ve soruları atla,
        başlıklar altındaki içeriği tek paragrafta birleştir.
        
        Args:
            text: İşlenecek ham metin
            
        Returns:
            İşlenmiş metin
        """
        logger.info("Gelişmiş metin ön işleme başlatılıyor")
        
        # Unicode normalizasyonu (NFKC)
        text = unicodedata.normalize('NFKC', text)
        
        # Satırlara böl
        lines = text.split('\n')
        
        # İşlenmiş metin için
        processed_paragraphs = []
        current_paragraph = []
        
        # Başlıklar altındaki içeriği toplamak için
        in_content_section = False
        
        for line in lines:
            line = line.strip()
            
            # Boş satırları atla
            if not line:
                # Bir paragraf biriktirilmişse, kaydet ve yeni bir paragrafa başla
                if current_paragraph:
                    processed_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                in_content_section = False
                continue
            
            # Başlık kontrolü
            if self.is_heading(line):
                # Bir paragraf biriktirilmişse, kaydet
                if current_paragraph:
                    processed_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Başlığı atla, ancak içerik bölümünde olduğumuzu işaretle
                in_content_section = True
                continue
            
            # Soru kontrolü
            if self.is_question(line):
                # Soruları atla
                continue
            
            # İçerik bölümündeyse ve geçerli bir satırsa, paragrafa ekle
            if in_content_section:
                # Sondaki boşlukları temizle
                line = line.strip()
                
                # Çok kısa satırları atla (muhtemelen alt başlıklar veya liste öğeleri)
                if len(line) < 20:
                    continue
                    
                current_paragraph.append(line)
        
        # Son paragrafı ekle
        if current_paragraph:
            processed_paragraphs.append(' '.join(current_paragraph))
        
        # Paragrafları birleştir
        processed_text = '\n\n'.join(processed_paragraphs)
        
        # Ardışık boşlukları tek bir boşlukla değiştir
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        # Ardışık satır sonlarını temizle
        processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text)
        
        logger.info("Gelişmiş metin ön işleme tamamlandı")
        return processed_text
    
    def preprocess_with_spacy(self, text: str) -> str:
        """
        spaCy kullanarak metin ön işleme.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            İşlenmiş metin
            
        Raises:
            ValueError: spaCy yüklü değilse veya model bulunamazsa
        """
        if not SPACY_AVAILABLE or self.nlp is None:
            raise ValueError(f"spaCy veya {self.language} dil modeli yüklü değil")
        
        logger.info("spaCy tabanlı metin ön işleme başlatılıyor")
        
        # Unicode normalizasyonu
        text = unicodedata.normalize('NFKC', text)
        
        # spaCy ile işle
        doc = self.nlp(text)
        
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
            
            current_paragraph.append(sent_text)
        
        # Son paragrafı ekle
        if current_paragraph:
            processed_paragraphs.append(' '.join(current_paragraph))
        
        # Paragrafları birleştir
        processed_text = '\n\n'.join(processed_paragraphs)
        
        logger.info("spaCy tabanlı metin ön işleme tamamlandı")
        return processed_text


class TextChunker:
    """Metni anlamlı parçalara bölme sınıfı."""
    
    def __init__(self, splitter_type: str = DEFAULT_TEXT_SPLITTER,
                 chunk_size: int = 2000, chunk_overlap: int = 200,
                 custom_separators: Optional[List[str]] = None):
        """
        Metin bölücüyü başlat.
        
        Args:
            splitter_type: Kullanılacak metin bölücü türü
            chunk_size: Her parçanın karakter cinsinden boyutu
            chunk_overlap: Parçalar arasındaki örtüşme (karakter cinsinden)
            custom_separators: Özel ayırıcılar listesi (isteğe bağlı)
        """
        self.splitter_type = splitter_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.custom_separators = custom_separators
        
        # Bölücü türünü doğrula
        if splitter_type not in TEXT_SPLITTER_TYPES:
            logger.warning(f"Belirtilen bölücü türü ({splitter_type}) bulunamadı. Varsayılan kullanılıyor.")
            self.splitter_type = DEFAULT_TEXT_SPLITTER
        
        self.splitter_class = TEXT_SPLITTER_TYPES[self.splitter_type]["class"]
    
    def chunk_text_semantic(self, text: str) -> List[Document]:
        """
        Metni anlamlı parçalara böl, paragraf bütünlüğünü koru.
        
        Args:
            text: Bölünecek metin
            
        Returns:
            Document nesneleri listesi
        """
        logger.info(f"Anlamsal metin bölümleme başlatılıyor (Bölücü: {self.splitter_type}, "
                    f"Boyut: {self.chunk_size}, Örtüşme: {self.chunk_overlap})")
        
        # Bölücü parametrelerini yapılandır
        if self.splitter_type == "recursive":
            # Varsayılan ayırıcılar
            separators = ["\n\n", "\n", ". ", " ", ""]
            
            # Özel ayırıcılar belirtilmişse kullan
            if self.custom_separators:
                separators = self.custom_separators + [""]
            
            text_splitter = self.splitter_class(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=separators
            )
        elif self.splitter_type == "sentence_transformers":
            # Karakter sayısı yerine token sayısı için ayarla
            token_chunk_size = max(1, self.chunk_size // 4)
            token_chunk_overlap = max(1, self.chunk_overlap // 4)
            
            text_splitter = self.splitter_class(
                chunk_size=token_chunk_size,
                chunk_overlap=token_chunk_overlap
            )
        else:  # token
            # Karakter sayısı yerine token sayısı için ayarla
            token_chunk_size = max(1, self.chunk_size // 4)
            token_chunk_overlap = max(1, self.chunk_overlap // 4)
            
            text_splitter = self.splitter_class(
                chunk_size=token_chunk_size,
                chunk_overlap=token_chunk_overlap
            )
        
        # Metni parçalara böl ve her parça için benzersiz ID oluştur
        chunks = text_splitter.create_documents([text])
        
        # Her parçaya benzersiz ID ekle
        for chunk in chunks:
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
        
        logger.info(f"Anlamsal metin bölümleme tamamlandı. {len(chunks)} parça oluşturuldu.")
        return chunks


class QuestionGenerator:
    """AI modelleri kullanarak metin parçalarından soru oluşturma sınıfı."""
    
    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: Optional[str] = None):
        """
        Soru oluşturucuyu başlat.
        
        Args:
            model: Kullanılacak OpenAI modeli
            system_prompt: Özel sistem mesajı (isteğe bağlı)
        """
        self.model = model
        self.system_prompt = system_prompt
        
        # Modeli doğrula
        if model not in AVAILABLE_MODELS:
            logger.warning(f"Belirtilen model ({model}) bulunamadı. Varsayılan kullanılıyor.")
            self.model = DEFAULT_MODEL
        
        self.model_info = AVAILABLE_MODELS[self.model]
        self.model_params = self.model_info["default_params"].copy()
    
    def generate_question(self, chunk_text: str, language: str = "en",
                        retries: int = 3, backoff_factor: float = 2.0) -> str:
        """
        Bir metin parçası için OpenAI API kullanarak soru oluştur.
        
        Args:
            chunk_text: Soru oluşturulacak metin parçası
            language: Metnin dili
            retries: Yeniden deneme sayısı
            backoff_factor: Yeniden denemeler için geri çekilme faktörü
            
        Returns:
            Oluşturulan soru
            
        Raises:
            Exception: API hatası durumunda
        """
        # Dile göre prompt ayarla
        if language == "tr":
            user_prompt = f"""
            Aşağıdaki metni oku ve bu metin hakkında en alakalı, açık uçlu tek bir soru oluştur.
            Soru, metindeki ana fikri veya önemli bir kavramı anlamaya yönelik olmalıdır.
            
            Metin:
            {chunk_text}
            
            Soru:
            """
        else:  # Varsayılan olarak İngilizce
            user_prompt = f"""
            Read the following text and create a single, relevant, open-ended question about it.
            The question should aim to understand the main idea or an important concept in the text.
            
            Text:
            {chunk_text}
            
            Question:
            """
        
        # Sistem mesajını ayarla
        if self.system_prompt:
            system_message = self.system_prompt
        else:
            if language == "tr":
                system_message = "Sen yardımcı bir eğitim asistanısın. Verilen metinler hakkında düşündürücü ve anlamlı sorular oluşturursun."
            else:
                system_message = "You are a helpful educational assistant. You create thoughtful and meaningful questions about given texts."
        
        # Test modunda sahte yanıt döndür
        if TEST_MODE:
            if language == "tr":
                return "Bu metin hakkında en önemli kavram nedir?"
            else:
                return "What is the most important concept in this text?"
        
        # API çağrısı için yeniden deneme mantığı
        for attempt in range(retries):
            try:
                # Jitter ekleyerek geri çekilme süresi hesapla
                if attempt > 0:
                    jitter = random.uniform(0.8, 1.2)
                    sleep_time = backoff_factor ** attempt * jitter
                    logger.info(f"{sleep_time:.2f} saniye bekleniyor...")
                    time.sleep(sleep_time)
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.model_params["temperature"],
                    top_p=self.model_params["top_p"],
                    max_tokens=self.model_params["max_tokens"]
                )
                
                question = response.choices[0].message.content.strip()
                return question
                
            except Exception as e:
                error_msg = f"API hatası (deneme {attempt+1}/{retries}): {str(e)}"
                logger.warning(error_msg)
                
                if attempt < retries - 1:
                    continue
                else:
                    logger.error(f"Maksimum yeniden deneme sayısına ulaşıldı. Son hata: {str(e)}")
                    if language == "tr":
                        return "API hatası nedeniyle soru oluşturulamadı."
                    else:
                        return "Could not generate question due to API error."
        
        # Bu noktaya ulaşılmamalı, ancak güvenlik için
        return "Question generation failed."


class DatasetCreator:
    """Metin parçaları ve sorulardan CSV veri setleri oluşturma sınıfı."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Veri seti oluşturucuyu başlat.
        
        Args:
            output_dir: CSV dosyaları için çıktı dizini
        """
        self.output_dir = output_dir
        
        # Çıktı dizini yoksa oluştur
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Çıktı dizini oluşturuldu: {output_dir}")
    
    def create_csv_dataset(self, chunks: List[Document], 
                          question_generator: QuestionGenerator,
                          pdf_name: str, language: str = "en",
                          system_prompt: Optional[str] = None,
                          max_workers: int = 1) -> str:
        """
        Metin parçaları ve oluşturulan sorulardan bir CSV veri seti oluştur.
        
        Args:
            chunks: Document nesneleri listesi
            question_generator: QuestionGenerator örneği
            pdf_name: PDF dosyasının adı
            language: Metnin dili
            system_prompt: Sistem sütunu için mesaj (isteğe bağlı)
            max_workers: Paralel işleme için maksimum iş parçacığı sayısı
            
        Returns:
            Oluşturulan CSV dosyasının yolu
        """
        logger.info(f"CSV veri seti oluşturma başlatılıyor (Model: {question_generator.model})")
        
        # Veri çerçevesi oluştur
        data = []
        
        # Soru oluşturma fonksiyonu
        def generate_question_for_chunk(chunk_with_index):
            index, chunk = chunk_with_index
            chunk_id = chunk.metadata["chunk_id"]
            user_text = chunk.page_content
            
            try:
                # Soru oluştur
                question = question_generator.generate_question(user_text, language)
                
                return {
                    "chunk_id": chunk_id,
                    "system": system_prompt or "",
                    "user": user_text,
                    "assistant": question
                }
            except Exception as e:
                logger.error(f"Parça {index} için soru oluşturma hatası: {str(e)}")
                # Hata durumunda boş soru
                if language == "tr":
                    error_question = "Soru oluşturma sırasında bir hata oluştu."
                else:
                    error_question = "An error occurred during question generation."
                
                return {
                    "chunk_id": chunk_id,
                    "system": system_prompt or "",
                    "user": user_text,
                    "assistant": error_question
                }
        
        # Paralel işleme kullan (max_workers > 1 ise)
        if max_workers > 1 and not TEST_MODE:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # tqdm ile ilerleme çubuğu
                futures = [executor.submit(generate_question_for_chunk, (i, chunk)) 
                          for i, chunk in enumerate(chunks)]
                
                for future in tqdm(as_completed(futures), total=len(chunks), desc="Sorular oluşturuluyor"):
                    data.append(future.result())
        else:
            # Seri işleme
            for i, chunk in enumerate(tqdm(chunks, desc="Sorular oluşturuluyor")):
                data.append(generate_question_for_chunk((i, chunk)))
        
        # DataFrame oluştur
        df = pd.DataFrame(data)
        
        # CSV dosya adı oluştur
        base_name = os.path.splitext(os.path.basename(pdf_name))[0]
        model_suffix = question_generator.model.replace("-", "_")
        csv_path = os.path.join(self.output_dir, f"{base_name}_{model_suffix}_dataset.csv")
        
        # CSV olarak kaydet
        df.to_csv(csv_path, index=False)
        
        logger.info(f"CSV veri seti oluşturma tamamlandı. Dosya: {csv_path}")
        return csv_path


class PDFToCSV:
    """PDF'den CSV dönüşüm işlemi için ana sınıf."""
    
    def __init__(self, input_dir: str = "input", output_dir: str = "output",
                model: str = DEFAULT_MODEL, splitter_type: str = DEFAULT_TEXT_SPLITTER,
                chunk_size: int = 2000, chunk_overlap: int = 200,
                language: str = DEFAULT_LANGUAGE, system_prompt: Optional[str] = None,
                custom_separators: Optional[List[str]] = None,
                max_workers: int = 1):
        """
        PDF'den CSV dönüşüm işlemini başlat.
        
        Args:
            input_dir: PDF dosyaları için giriş dizini
            output_dir: CSV dosyaları için çıktı dizini
            model: Kullanılacak OpenAI modeli
            splitter_type: Metin bölücü algoritması
            chunk_size: Metin parçalarının boyutu
            chunk_overlap: Parçalar arasındaki örtüşme
            language: Belgenin dili
            system_prompt: Sistem sütunu için mesaj (isteğe bağlı)
            custom_separators: Özel ayırıcılar listesi (isteğe bağlı)
            max_workers: Paralel işleme için maksimum iş parçacığı sayısı
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.splitter_type = splitter_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.system_prompt = system_prompt
        self.custom_separators = custom_separators
        self.max_workers = max_workers
        
        # Bileşenleri başlat
        self.pdf_processor = PDFProcessor(input_dir, language)
        self.question_generator = QuestionGenerator(model, system_prompt)
        self.dataset_creator = DatasetCreator(output_dir)
    
    def process_pdf(self, specific_file: Optional[str] = None) -> Tuple[str, Dict[str, Any], int, str]:
        """
        Bir PDF dosyasını işle ve bir CSV veri seti oluştur.
        
        Args:
            specific_file: İşlenecek belirli bir PDF dosyası (isteğe bağlı)
            
        Returns:
            Tuple[str, Dict[str, Any], int, str]: 
                - Oluşturulan CSV dosyasının yolu
                - İstatistikler
                - Parça sayısı
                - Algılanan dil
            
        Raises:
            Exception: İşlem sırasında hata oluşursa
        """
        # PDF yolunu al
        pdf_path = self.pdf_processor.get_pdf_path(specific_file)
        pdf_name = os.path.basename(pdf_path)
        
        # PDF'den metin çıkar
        text, stats = self.pdf_processor.extract_text_from_pdf(pdf_path)
        
        # Dili algıla (auto olarak ayarlanmışsa)
        detected_language = self.pdf_processor.detect_language(text)
        
        # Metin işleyiciyi algılanan dille başlat
        text_processor = TextProcessor(detected_language)
        
        # Metni ön işle
        if SPACY_AVAILABLE and text_processor.nlp is not None:
            try:
                processed_text = text_processor.preprocess_with_spacy(text)
                logger.info("Metin spaCy ile işlendi")
            except Exception as e:
                logger.warning(f"spaCy ile işleme hatası: {str(e)}. Gelişmiş metin işleme kullanılıyor.")
                processed_text = text_processor.preprocess_text_advanced(text)
        else:
            processed_text = text_processor.preprocess_text_advanced(text)
        
        # Metin bölücüyü başlat
        text_chunker = TextChunker(
            self.splitter_type, 
            self.chunk_size, 
            self.chunk_overlap,
            self.custom_separators
        )
        
        # Metni parçalara böl
        chunks = text_chunker.chunk_text_semantic(processed_text)
        
        # CSV veri seti oluştur
        csv_path = self.dataset_creator.create_csv_dataset(
            chunks, 
            self.question_generator, 
            pdf_name, 
            detected_language,
            self.system_prompt,
            self.max_workers
        )
        
        # İstatistikleri ve sonuçları döndür
        return csv_path, stats, len(chunks), detected_language
    
    def process_batch(self, max_files: Optional[int] = None) -> List[Tuple[str, str, int, str]]:
        """
        Giriş dizinindeki birden çok PDF dosyasını işle.
        
        Args:
            max_files: İşlenecek maksimum dosya sayısı (isteğe bağlı)
            
        Returns:
            List[Tuple[str, str, int, str]]: Her dosya için sonuçlar listesi
                - PDF dosya adı
                - CSV dosya yolu
                - Parça sayısı
                - Algılanan dil
        """
        # Giriş dizinindeki tüm PDF dosyalarını bul
        pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        results = []
        
        for pdf_file in tqdm(pdf_files, desc="PDF dosyaları işleniyor"):
            try:
                logger.info(f"İşleniyor: {pdf_file}")
                csv_path, stats, chunk_count, detected_language = self.process_pdf(pdf_file)
                results.append((pdf_file, csv_path, chunk_count, detected_language))
                logger.info(f"Tamamlandı: {pdf_file} -> {csv_path}")
            except Exception as e:
                logger.error(f"{pdf_file} dosyası işlenirken hata: {str(e)}")
        
        return results


def list_available_models():
    """Kullanılabilir OpenAI modellerini listele."""
    print("\n📋 Kullanılabilir OpenAI Modelleri:")
    print("=" * 80)
    print(f"{'Model ID':<20} {'Model Adı':<20} {'Açıklama'}")
    print("-" * 80)
    
    for model_id, model_info in AVAILABLE_MODELS.items():
        print(f"{model_id:<20} {model_info['name']:<20} {model_info['description']}")
    
    print("=" * 80)

def list_text_splitters():
    """Kullanılabilir metin bölücü algoritmalarını listele."""
    print("\n📋 Kullanılabilir Metin Bölücü Algoritmaları:")
    print("=" * 80)
    print(f"{'Bölücü ID':<25} {'Bölücü Adı':<30} {'Açıklama'}")
    print("-" * 80)
    
    for splitter_id, splitter_info in TEXT_SPLITTER_TYPES.items():
        print(f"{splitter_id:<25} {splitter_info['name']:<30} {splitter_info['description']}")
    
    print("=" * 80)

def list_supported_languages():
    """Desteklenen dilleri listele."""
    print("\n📋 Desteklenen Diller:")
    print("=" * 60)
    print(f"{'Dil Kodu':<15} {'Dil Adı'}")
    print("-" * 60)
    
    print(f"{'auto':<15} {'Otomatik algılama'}")
    for lang_code, lang_info in SUPPORTED_LANGUAGES.items():
        print(f"{lang_code:<15} {lang_info['name']}")
    
    print("=" * 60)

def show_version_info():
    """Versiyon bilgisini göster."""
    print(f"CorpusCrafter: PDF to AI Dataset Generator v{__version__}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Bağımlılık sürümlerini kontrol et
    dependencies = {
        "langchain": "langchain",
        "PyPDF2": "PyPDF2",
        "pandas": "pandas",
        "openai": "openai"
    }
    
    for dep_name, package_name in dependencies.items():
        try:
            version = importlib.metadata.version(package_name)
            print(f"{dep_name}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{dep_name}: Yüklü değil")
    
    # İsteğe bağlı bağımlılıkları kontrol et
    optional_deps = {
        "langdetect": LANGDETECT_AVAILABLE,
        "spacy": SPACY_AVAILABLE
    }
    
    print("\nİsteğe Bağlı Bağımlılıklar:")
    for dep_name, installed in optional_deps.items():
        status = "Yüklü" if installed else "Yüklü değil"
        print(f"{dep_name}: {status}")

def main():
    """Komut satırı kullanımı için ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description="CorpusCrafter: PDF to AI Dataset Generator - Gelişmiş Metin İşleme",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", default="input", help="PDF dosyaları için giriş dizini")
    parser.add_argument("--output", "-o", default="output", help="CSV dosyaları için çıktı dizini")
    parser.add_argument("--file", "-f", help="İşlenecek belirli bir PDF dosyası (isteğe bağlı)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, 
                        help="Kullanılacak OpenAI modeli")
    parser.add_argument("--splitter", "-s", default=DEFAULT_TEXT_SPLITTER,
                        help="Kullanılacak metin bölücü algoritması")
    parser.add_argument("--chunk-size", "-cs", type=int, default=2000,
                        help="Metin parçalarının karakter cinsinden boyutu")
    parser.add_argument("--chunk-overlap", "-co", type=int, default=200,
                        help="Parçalar arasındaki karakter cinsinden örtüşme")
    parser.add_argument("--language", "-l", default=DEFAULT_LANGUAGE,
                        help="Belgenin birincil dili (otomatik algılama için 'auto')")
    parser.add_argument("--system-prompt", "-sp", 
                        help="Sistem sütunu için özel mesaj")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Giriş dizinindeki tüm PDF dosyalarını işle")
    parser.add_argument("--max-files", "-mf", type=int,
                        help="Batch modunda işlenecek maksimum dosya sayısı")
    parser.add_argument("--custom-separators", "-sep", nargs="+",
                        help="Metin bölme için özel ayırıcılar")
    parser.add_argument("--max-workers", "-w", type=int, default=1,
                        help="Paralel işleme için maksimum iş parçacığı sayısı")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Ayrıntılı günlük kaydını etkinleştir")
    parser.add_argument("--log-file", 
                        help="Günlük dosyasının adı")
    parser.add_argument("--test-mode", "-t", action="store_true",
                        help="Test modunu etkinleştir (OpenAI API çağrıları yapılmaz)")
    
    # Bilgi komutları
    parser.add_argument("--list-models", "-lm", action="store_true",
                        help="Kullanılabilir OpenAI modellerini listele")
    parser.add_argument("--list-splitters", "-ls", action="store_true",
                        help="Kullanılabilir metin bölücü algoritmalarını listele")
    parser.add_argument("--list-languages", "-ll", action="store_true",
                        help="Desteklenen dilleri listele")
    parser.add_argument("--version", action="store_true",
                        help="Versiyon bilgisini göster")
    
    args = parser.parse_args()
    
    # Bilgi komutlarını işle
    if args.list_models:
        list_available_models()
        return
    
    if args.list_splitters:
        list_text_splitters()
        return
    
    if args.list_languages:
        list_supported_languages()
        return
    
    if args.version:
        show_version_info()
        return
    
    # Günlük seviyesini ayarla
    if args.verbose:
        logger = configure_logging(
            log_file=args.log_file or "pdf_to_csv.log",
            log_level=logging.DEBUG
        )
    
    # Test modunu ayarla
    if args.test_mode:
        global TEST_MODE
        TEST_MODE = True
        os.environ["CORPUSCRAFTER_TEST_MODE"] = "true"
        logger.info("Test modu etkinleştirildi")
    
    # PDF'den CSV dönüşüm işlemini başlat
    pdf_to_csv = PDFToCSV(
        input_dir=args.input,
        output_dir=args.output,
        model=args.model,
        splitter_type=args.splitter,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        language=args.language,
        system_prompt=args.system_prompt,
        custom_separators=args.custom_separators,
        max_workers=args.max_workers
    )
    
    try:
        if args.batch:
            # Batch işleme
            results = pdf_to_csv.process_batch(args.max_files)
            print(f"\n✅ {len(results)} PDF dosyası başarıyla işlendi.")
            
            # Sonuçları göster
            if results:
                print("\nİşleme Sonuçları:")
                print("-" * 80)
                print(f"{'PDF Dosyası':<30} {'CSV Dosyası':<30} {'Parça Sayısı':<15} {'Dil'}")
                print("-" * 80)
                
                for pdf_file, csv_path, chunk_count, language in results:
                    csv_name = os.path.basename(csv_path)
                    print(f"{pdf_file:<30} {csv_name:<30} {chunk_count:<15} {language}")
        else:
            # Tek dosya işleme
            csv_path, stats, chunk_count, detected_language = pdf_to_csv.process_pdf(args.file)
            
            print(f"\n✅ PDF başarıyla işlendi!")
            print(f"📊 İstatistikler:")
            print(f"  - Toplam sayfa: {stats['total_pages']}")
            print(f"  - Atlanan sayfa: {stats['skipped_pages']}")
            print(f"  - Boş sayfa: {stats['empty_pages']}")
            print(f"  - Görüntü sayfası: {stats['image_pages']}")
            print(f"  - Oluşturulan parça: {chunk_count}")
            print(f"  - Algılanan dil: {detected_language}")
            print(f"  - CSV dosyası: {csv_path}")
    
    except Exception as e:
        logger.error(f"İşlem hatası: {str(e)}")
        print(f"\n❌ Hata: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
