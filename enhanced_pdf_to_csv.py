"""
CorpusCrafter: PDF'ten AI Veri Seti Oluşturucu - Entegrasyon Modülü
-----------------------------------------
Bu modül, gelişmiş özelliklerin ana pipeline'a entegrasyonunu sağlar.
Gelişmiş Soru Üretimi, Gelişmiş NLP Entegrasyonu ve Akıllı Metin Bölümleme
modüllerini mevcut PDF'ten CSV'ye dönüştürme sürecine entegre eder.
"""

import os
import sys
import logging
import uuid
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path

# Mevcut modülü içe aktar
sys.path.append('/home/ubuntu')
from fixed_pdf_to_csv_core import (
    PDFToCSV,
    PDFProcessor,
    TextProcessor,
    TextChunker,
    QuestionGenerator,
    DatasetCreator,
    logger,
    DEFAULT_MODEL,
    DEFAULT_TEXT_SPLITTER,
    DEFAULT_LANGUAGE
)

# Yeni modülleri içe aktar
try:
    from advanced_question_generator import AdvancedQuestionGenerator
    ADVANCED_QUESTION_GENERATOR_AVAILABLE = True
except ImportError:
    ADVANCED_QUESTION_GENERATOR_AVAILABLE = False
    logger.warning("Gelişmiş Soru Üretimi modülü bulunamadı. Standart soru üretimi kullanılacak.")

try:
    from advanced_nlp_processor import AdvancedNLPProcessor
    ADVANCED_NLP_PROCESSOR_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_PROCESSOR_AVAILABLE = False
    logger.warning("Gelişmiş NLP Entegrasyonu modülü bulunamadı. Standart metin işleme kullanılacak.")

try:
    from smart_text_chunker import SmartTextChunker
    SMART_TEXT_CHUNKER_AVAILABLE = True
except ImportError:
    SMART_TEXT_CHUNKER_AVAILABLE = False
    logger.warning("Akıllı Metin Bölümleme modülü bulunamadı. Standart metin bölümleme kullanılacak.")


class EnhancedPDFToCSV(PDFToCSV):
    """Gelişmiş özelliklere sahip PDF'ten CSV dönüştürücü."""
    
    def __init__(self, 
                 input_dir: str = "input", 
                 output_dir: str = "output",
                 model: str = DEFAULT_MODEL, 
                 splitter_type: str = DEFAULT_TEXT_SPLITTER,
                 chunk_size: int = 2000, 
                 chunk_overlap: int = 200,
                 language: str = DEFAULT_LANGUAGE, 
                 system_prompt: Optional[str] = None,
                 custom_separators: Optional[List[str]] = None,
                 max_workers: int = 1,
                 # Gelişmiş Soru Üretimi parametreleri
                 use_advanced_question_generator: bool = False,
                 question_types: Optional[List[str]] = None,
                 question_difficulty: str = "mixed",
                 question_diversity: float = 0.7,
                 question_count_per_chunk: int = 3,
                 question_model: Optional[str] = None,
                 # Gelişmiş NLP Entegrasyonu parametreleri
                 use_advanced_nlp: bool = False,
                 nlp_model_type: str = "transformer",
                 nlp_model_name: Optional[str] = None,
                 enable_ner: bool = True,
                 enable_sentiment: bool = True,
                 enable_summarization: bool = True,
                 enable_keyword_extraction: bool = True,
                 # Akıllı Metin Bölümleme parametreleri
                 use_smart_chunking: bool = False,
                 semantic_chunking: bool = False,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 clustering_method: str = "kmeans",
                 n_clusters: Optional[int] = None,
                 # Genel parametreler
                 use_gpu: bool = False,
                 cache_dir: Optional[str] = None):
        """
        Gelişmiş PDF'ten CSV dönüşüm işlemini başlat.
        
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
            
            # Gelişmiş Soru Üretimi parametreleri
            use_advanced_question_generator: Gelişmiş soru üretimini etkinleştir
            question_types: Üretilecek soru türleri listesi
            question_difficulty: Soru zorluk seviyesi ('easy', 'medium', 'hard', 'mixed')
            question_diversity: Soru çeşitliliği (0-1 arası)
            question_count_per_chunk: Her parça için üretilecek soru sayısı
            question_model: Soru üretimi için özel model
            
            # Gelişmiş NLP Entegrasyonu parametreleri
            use_advanced_nlp: Gelişmiş NLP işlemeyi etkinleştir
            nlp_model_type: NLP model türü ('spacy', 'transformer', 'stanza', 'flair')
            nlp_model_name: Kullanılacak özel model adı
            enable_ner: Varlık ismi tanıma işlemini etkinleştir
            enable_sentiment: Duygu analizi işlemini etkinleştir
            enable_summarization: Metin özetleme işlemini etkinleştir
            enable_keyword_extraction: Anahtar kelime çıkarma işlemini etkinleştir
            
            # Akıllı Metin Bölümleme parametreleri
            use_smart_chunking: Akıllı metin bölümlemeyi etkinleştir
            semantic_chunking: Semantik bölümleme kullan
            embedding_model: Cümle gömme modeli
            clustering_method: Kümeleme metodu
            n_clusters: Küme sayısı
            
            # Genel parametreler
            use_gpu: GPU kullanımını etkinleştir
            cache_dir: Model önbelleği için dizin
        """
        # Temel sınıf başlatma
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language=language,
            system_prompt=system_prompt,
            custom_separators=custom_separators,
            max_workers=max_workers
        )
        
        # Gelişmiş özellik parametreleri
        self.use_advanced_question_generator = use_advanced_question_generator
        self.question_types = question_types
        self.question_difficulty = question_difficulty
        self.question_diversity = question_diversity
        self.question_count_per_chunk = question_count_per_chunk
        self.question_model = question_model or model
        
        self.use_advanced_nlp = use_advanced_nlp
        self.nlp_model_type = nlp_model_type
        self.nlp_model_name = nlp_model_name
        self.enable_ner = enable_ner
        self.enable_sentiment = enable_sentiment
        self.enable_summarization = enable_summarization
        self.enable_keyword_extraction = enable_keyword_extraction
        
        self.use_smart_chunking = use_smart_chunking
        self.semantic_chunking = semantic_chunking
        self.embedding_model = embedding_model
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        
        # Bileşenleri yeniden başlat
        self._initialize_components()
    
    def _initialize_components(self):
        """Gelişmiş bileşenleri başlat."""
        # PDF işleyici
        self.pdf_processor = PDFProcessor(self.input_dir, self.language)
        
        # Metin işleyici
        if self.use_advanced_nlp and ADVANCED_NLP_PROCESSOR_AVAILABLE:
            self.text_processor = AdvancedNLPProcessor(
                language=self.language,
                model_type=self.nlp_model_type,
                model_name=self.nlp_model_name,
                use_gpu=self.use_gpu,
                enable_ner=self.enable_ner,
                enable_sentiment=self.enable_sentiment,
                enable_summarization=self.enable_summarization,
                enable_keyword_extraction=self.enable_keyword_extraction,
                cache_dir=self.cache_dir
            )
            logger.info("Gelişmiş NLP işleyici başlatıldı")
        else:
            self.text_processor = TextProcessor(self.language)
            logger.info("Standart metin işleyici başlatıldı")
        
        # Metin bölücü
        if self.use_smart_chunking and SMART_TEXT_CHUNKER_AVAILABLE:
            self.text_chunker = SmartTextChunker(
                splitter_type=self.splitter_type,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                custom_separators=self.custom_separators,
                semantic_chunking=self.semantic_chunking,
                embedding_model=self.embedding_model,
                clustering_method=self.clustering_method,
                n_clusters=self.n_clusters,
                use_gpu=self.use_gpu,
                language=self.language,
                cache_dir=self.cache_dir
            )
            logger.info("Akıllı metin bölücü başlatıldı")
        else:
            self.text_chunker = TextChunker(
                splitter_type=self.splitter_type,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                custom_separators=self.custom_separators
            )
            logger.info("Standart metin bölücü başlatıldı")
        
        # Soru üretici
        if self.use_advanced_question_generator and ADVANCED_QUESTION_GENERATOR_AVAILABLE:
            self.question_generator = AdvancedQuestionGenerator(
                model=self.question_model,
                system_prompt=self.system_prompt,
                question_types=self.question_types,
                difficulty=self.question_difficulty,
                diversity=self.question_diversity,
                questions_per_chunk=self.question_count_per_chunk,
                language=self.language
            )
            logger.info("Gelişmiş soru üretici başlatıldı")
        else:
            self.question_generator = QuestionGenerator(
                model=self.model,
                system_prompt=self.system_prompt
            )
            logger.info("Standart soru üretici başlatıldı")
        
        # Veri seti oluşturucu
        self.dataset_creator = DatasetCreator(self.output_dir)
    
    def process_pdf(self, pdf_path: str) -> str:
        """
        PDF dosyasını işle ve CSV'ye dönüştür.
        
        Args:
            pdf_path: İşlenecek PDF dosyasının yolu
            
        Returns:
            Oluşturulan CSV dosyasının yolu
        """
        logger.info(f"PDF işleniyor: {pdf_path}")
        
        # PDF'ten metni çıkar
        text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        
        # Metni işle
        if self.use_advanced_nlp and ADVANCED_NLP_PROCESSOR_AVAILABLE:
            processed_text = self.text_processor.preprocess_with_advanced_nlp(text)
        else:
            processed_text = self.text_processor.preprocess_text(text)
        
        # Metni parçalara böl
        if self.use_smart_chunking and SMART_TEXT_CHUNKER_AVAILABLE:
            if self.semantic_chunking:
                chunks = self.text_chunker.chunk_text(processed_text)
            else:
                # Alternatif bölümleme yöntemleri
                if self.clustering_method == "textrank":
                    chunks = self.text_chunker.chunk_text_with_textrank(processed_text)
                elif self.clustering_method == "topic":
                    chunks = self.text_chunker.chunk_text_with_topic_modeling(processed_text)
                elif self.clustering_method == "hierarchical":
                    chunks = self.text_chunker.chunk_text_with_hierarchical(processed_text)
                elif self.clustering_method == "sliding_window":
                    chunks = self.text_chunker.chunk_text_with_sliding_window(processed_text)
                else:
                    chunks = self.text_chunker.chunk_text(processed_text)
        else:
            chunks = self.text_chunker.chunk_text(processed_text)
        
        # Sorular oluştur
        if self.use_advanced_question_generator and ADVANCED_QUESTION_GENERATOR_AVAILABLE:
            # Gelişmiş soru üretimi
            questions = []
            for chunk in chunks:
                chunk_text = chunk["chunk_text"] if "chunk_text" in chunk else chunk["text"]
                chunk_questions = self.question_generator.generate_diverse_questions(chunk_text)
                questions.extend(chunk_questions)
        else:
            # Standart soru üretimi
            questions = []
            for chunk in chunks:
                chunk_text = chunk["chunk_text"] if "chunk_text" in chunk else chunk["text"]
                chunk_question = self.question_generator.generate_question(chunk_text)
                questions.append(chunk_question)
        
        # CSV veri seti oluştur
        pdf_name = os.path.basename(pdf_path)
        csv_path = self.dataset_creator.create_dataset(pdf_name, chunks, questions)
        
        logger.info(f"CSV oluşturuldu: {csv_path}")
        return csv_path
    
    def process_all_pdfs(self) -> List[str]:
        """
        Tüm PDF dosyalarını işle.
        
        Returns:
            Oluşturulan CSV dosyalarının yolları
        """
        return super().process_all_pdfs()
    
    def get_feature_status(self) -> Dict[str, bool]:
        """
        Gelişmiş özelliklerin durumunu döndür.
        
        Returns:
            Özellik durumları sözlüğü
        """
        return {
            "advanced_question_generator": self.use_advanced_question_generator and ADVANCED_QUESTION_GENERATOR_AVAILABLE,
            "advanced_nlp": self.use_advanced_nlp and ADVANCED_NLP_PROCESSOR_AVAILABLE,
            "smart_chunking": self.use_smart_chunking and SMART_TEXT_CHUNKER_AVAILABLE,
            "semantic_chunking": self.semantic_chunking and SMART_TEXT_CHUNKER_AVAILABLE,
            "gpu_support": self.use_gpu
        }


# Test kodu
if __name__ == "__main__":
    # Gelişmiş özelliklerin durumunu kontrol et
    print("Gelişmiş özelliklerin durumu:")
    print(f"  Gelişmiş Soru Üretimi: {'Kullanılabilir' if ADVANCED_QUESTION_GENERATOR_AVAILABLE else 'Kullanılamaz'}")
    print(f"  Gelişmiş NLP Entegrasyonu: {'Kullanılabilir' if ADVANCED_NLP_PROCESSOR_AVAILABLE else 'Kullanılamaz'}")
    print(f"  Akıllı Metin Bölümleme: {'Kullanılabilir' if SMART_TEXT_CHUNKER_AVAILABLE else 'Kullanılamaz'}")
    
    # Test için gelişmiş dönüştürücüyü başlat
    converter = EnhancedPDFToCSV(
        input_dir="input",
        output_dir="output",
        model="gpt-4o-mini",
        chunk_size=1500,
        chunk_overlap=150,
        language="tr",
        # Gelişmiş özellikler
        use_advanced_question_generator=ADVANCED_QUESTION_GENERATOR_AVAILABLE,
        question_types=["factual", "conceptual", "analytical"],
        question_difficulty="mixed",
        
        use_advanced_nlp=ADVANCED_NLP_PROCESSOR_AVAILABLE,
        nlp_model_type="transformer",
        
        use_smart_chunking=SMART_TEXT_CHUNKER_AVAILABLE,
        semantic_chunking=True,
        clustering_method="kmeans"
    )
    
    # Özellik durumunu göster
    feature_status = converter.get_feature_status()
    print("\nEtkin özellikler:")
    for feature, status in feature_status.items():
        print(f"  {feature}: {'Etkin' if status else 'Devre dışı'}")
    
    # Test PDF'i varsa işle
    import glob
    pdf_files = glob.glob("input/*.pdf")
    if pdf_files:
        print(f"\nTest PDF dosyası işleniyor: {pdf_files[0]}")
        csv_path = converter.process_pdf(pdf_files[0])
        print(f"CSV oluşturuldu: {csv_path}")
    else:
        print("\nTest için PDF dosyası bulunamadı. Lütfen 'input' dizinine bir PDF dosyası ekleyin.")
