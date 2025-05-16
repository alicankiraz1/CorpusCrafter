"""
CorpusCrafter: Akıllı Metin Bölümleme Modülü
-----------------------------------------
Bu modül, PDF'ten CSV veri seti oluşturucu için gelişmiş metin bölümleme işlevselliği sağlar.
Transformer tabanlı modeller, cümle gömme vektörleri ve gelişmiş segmentasyon algoritmaları kullanarak
metni daha anlamlı ve tutarlı parçalara böler.
"""

import os
import logging
import re
import uuid
import time
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path

# Mevcut modülü içe aktar
import sys
sys.path.append('/home/ubuntu')
from fixed_pdf_to_csv_core import (
    TextChunker, 
    logger, 
    DEFAULT_TEXT_SPLITTER
)

# Bağımlılık kontrolü
def check_dependencies():
    """Gerekli bağımlılıkları kontrol et ve durumlarını döndür."""
    dependencies = {
        "sentence_transformers": False,
        "sklearn": False,
        "nltk": False,
        "torch": False,
        "numpy": False,
        "gensim": False,
        "networkx": False
    }
    
    # sentence_transformers
    try:
        import sentence_transformers
        dependencies["sentence_transformers"] = True
    except ImportError:
        pass
    
    # sklearn
    try:
        import sklearn
        dependencies["sklearn"] = True
    except ImportError:
        pass
    
    # nltk
    try:
        import nltk
        dependencies["nltk"] = True
    except ImportError:
        pass
    
    # torch
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    # numpy
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        pass
    
    # gensim
    try:
        import gensim
        dependencies["gensim"] = True
    except ImportError:
        pass
    
    # networkx
    try:
        import networkx
        dependencies["networkx"] = True
    except ImportError:
        pass
    
    return dependencies


class SmartTextChunker(TextChunker):
    """Akıllı metin bölümleme için genişletilmiş sınıf."""
    
    def __init__(self, 
                 splitter_type: str = DEFAULT_TEXT_SPLITTER,
                 chunk_size: int = 2000, 
                 chunk_overlap: int = 200,
                 custom_separators: Optional[List[str]] = None,
                 semantic_chunking: bool = False,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 clustering_method: str = "kmeans",
                 n_clusters: Optional[int] = None,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 3000,
                 similarity_threshold: float = 0.75,
                 use_gpu: bool = False,
                 language: str = "en",
                 preserve_paragraph: bool = True,
                 preserve_sentence: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Akıllı metin bölücüyü başlat.
        
        Args:
            splitter_type: Kullanılacak metin bölücü türü
            chunk_size: Her parçanın karakter cinsinden boyutu
            chunk_overlap: Parçalar arasındaki örtüşme (karakter cinsinden)
            custom_separators: Özel ayırıcılar listesi (isteğe bağlı)
            semantic_chunking: Semantik bölümleme kullan
            embedding_model: Cümle gömme modeli
            clustering_method: Kümeleme metodu ('kmeans', 'agglomerative', 'dbscan', 'spectral', 'semantic_similarity')
            n_clusters: Küme sayısı (None ise otomatik belirlenir)
            min_chunk_size: Minimum parça boyutu (karakter cinsinden)
            max_chunk_size: Maksimum parça boyutu (karakter cinsinden)
            similarity_threshold: Benzerlik eşiği (0-1 arası)
            use_gpu: GPU kullanımını etkinleştir
            language: Metin dili
            preserve_paragraph: Paragraf bütünlüğünü koru
            preserve_sentence: Cümle bütünlüğünü koru
            cache_dir: Model önbelleği için dizin
        """
        super().__init__(splitter_type, chunk_size, chunk_overlap, custom_separators)
        self.semantic_chunking = semantic_chunking
        self.embedding_model = embedding_model
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.use_gpu = use_gpu
        self.language = language
        self.preserve_paragraph = preserve_paragraph
        self.preserve_sentence = preserve_sentence
        self.cache_dir = cache_dir
        
        # Bağımlılıkları kontrol et
        self.dependencies = check_dependencies()
        
        # Modelleri yükle
        self.model = None
        if self.semantic_chunking:
            self._load_models()
    
    def _load_models(self):
        """Semantik bölümleme için modelleri yükle."""
        # Gerekli bağımlılıkları kontrol et
        if not self.dependencies["sentence_transformers"]:
            logger.warning("sentence_transformers yüklü değil. pip install sentence-transformers komutuyla yükleyin.")
            self.semantic_chunking = False
            return
        
        if not self.dependencies["sklearn"]:
            logger.warning("scikit-learn yüklü değil. pip install scikit-learn komutuyla yükleyin.")
            self.semantic_chunking = False
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # GPU kullanımını yapılandır
            device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
            
            # Cümle gömme modelini yükle
            self.model = SentenceTransformer(self.embedding_model, device=device, cache_folder=self.cache_dir)
            logger.info(f"Cümle gömme modeli {self.embedding_model} başarıyla yüklendi")
            
            # NLTK kaynaklarını indir
            if self.dependencies["nltk"]:
                import nltk
                nltk.download('punkt', quiet=True)
                
                # Dile özgü punkt modelini indir
                try:
                    nltk_lang = self._map_language_code_to_nltk(self.language)
                    nltk.download(f'punkt/{nltk_lang}.pickle', quiet=True)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")
            self.semantic_chunking = False
    
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
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Metni parçalara böl.
        
        Args:
            text: Bölünecek metin
            
        Returns:
            Parça sözlükleri listesi
        """
        if self.semantic_chunking and all([
            self.dependencies["sentence_transformers"],
            self.dependencies["sklearn"],
            self.dependencies["numpy"]
        ]):
            logger.info("Semantik metin bölümleme başlatılıyor")
            return self._chunk_text_semantic_advanced(text)
        else:
            logger.info("Standart metin bölümleme kullanılıyor")
            return super().chunk_text(text)
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Metni paragraflara böl."""
        # Boş satırları kullanarak paragraflara böl
        paragraphs = text.split('\n\n')
        
        # Tek satırlık boşlukları da kontrol et
        result = []
        for p in paragraphs:
            # İç paragrafları böl
            inner_paragraphs = p.split('\n')
            # Boş olmayan paragrafları ekle
            for inner_p in inner_paragraphs:
                if inner_p.strip():
                    result.append(inner_p.strip())
        
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Metni cümlelere böl."""
        if self.dependencies["nltk"]:
            import nltk
            try:
                # Dile özgü tokenizer kullan
                nltk_lang = self._map_language_code_to_nltk(self.language)
                sentences = nltk.sent_tokenize(text, language=nltk_lang)
            except:
                # Fallback olarak varsayılan tokenizer kullan
                sentences = nltk.sent_tokenize(text)
        else:
            # Basit cümle bölme
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Boş cümleleri filtrele
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_text_semantic_advanced(self, text: str) -> List[Dict[str, str]]:
        """
        Gelişmiş semantik bölümleme ile metni anlamlı parçalara böl.
        
        Args:
            text: Bölünecek metin
            
        Returns:
            Parça sözlükleri listesi
        """
        import numpy as np
        
        # Metni cümlelere ve paragraflara böl
        paragraphs = self._split_into_paragraphs(text)
        
        # Paragrafları cümlelere böl
        sentences_by_paragraph = [self._split_into_sentences(p) for p in paragraphs]
        
        # Tüm cümleleri düzleştir
        all_sentences = [s for sentences in sentences_by_paragraph for s in sentences]
        
        # Cümle sayısı çok azsa standart bölümleme kullan
        if len(all_sentences) < 5:
            logger.info("Cümle sayısı çok az, standart bölümleme kullanılıyor")
            return super().chunk_text(text)
        
        # Cümle gömme vektörlerini hesapla
        sentence_embeddings = self.model.encode(all_sentences, show_progress_bar=False)
        
        # Seçilen kümeleme metoduna göre cümleleri grupla
        if self.clustering_method == "kmeans":
            chunks = self._cluster_kmeans(all_sentences, sentence_embeddings)
        elif self.clustering_method == "agglomerative":
            chunks = self._cluster_agglomerative(all_sentences, sentence_embeddings)
        elif self.clustering_method == "dbscan":
            chunks = self._cluster_dbscan(all_sentences, sentence_embeddings)
        elif self.clustering_method == "spectral":
            chunks = self._cluster_spectral(all_sentences, sentence_embeddings)
        elif self.clustering_method == "semantic_similarity":
            chunks = self._cluster_semantic_similarity(all_sentences, sentence_embeddings)
        else:
            logger.warning(f"Bilinmeyen kümeleme metodu: {self.clustering_method}, kmeans kullanılıyor")
            chunks = self._cluster_kmeans(all_sentences, sentence_embeddings)
        
        # Parça boyutlarını kontrol et ve gerekirse birleştir veya böl
        chunks = self._optimize_chunk_sizes(chunks)
        
        # Parça sözlükleri oluştur
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Boş parçaları atla
                chunk_dict = {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "source": f"semantic_chunking_{self.clustering_method}"
                }
                chunk_dicts.append(chunk_dict)
        
        logger.info(f"Semantik metin bölümleme tamamlandı. {len(chunk_dicts)} parça oluşturuldu.")
        return chunk_dicts
    
    def _cluster_kmeans(self, sentences: List[str], embeddings) -> List[str]:
        """K-means kümeleme ile cümleleri grupla."""
        from sklearn.cluster import KMeans
        
        # Küme sayısını belirle
        n_clusters = self.n_clusters
        if n_clusters is None:
            # Otomatik küme sayısı belirleme
            n_clusters = max(2, min(len(sentences) // 5, len(sentences) // (self.chunk_size // 100)))
        
        # K-means kümeleme
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Kümelere göre cümleleri grupla
        grouped_sentences = [[] for _ in range(n_clusters)]
        for i, cluster_id in enumerate(clusters):
            grouped_sentences[cluster_id].append(sentences[i])
        
        # Her küme için cümleleri birleştir
        chunks = [' '.join(group) for group in grouped_sentences if group]
        
        return chunks
    
    def _cluster_agglomerative(self, sentences: List[str], embeddings) -> List[str]:
        """Aglomeratif kümeleme ile cümleleri grupla."""
        from sklearn.cluster import AgglomerativeClustering
        
        # Küme sayısını belirle
        n_clusters = self.n_clusters
        if n_clusters is None:
            # Otomatik küme sayısı belirleme
            n_clusters = max(2, min(len(sentences) // 5, len(sentences) // (self.chunk_size // 100)))
        
        # Aglomeratif kümeleme
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clustering.fit_predict(embeddings)
        
        # Kümelere göre cümleleri grupla
        grouped_sentences = [[] for _ in range(n_clusters)]
        for i, cluster_id in enumerate(clusters):
            grouped_sentences[cluster_id].append(sentences[i])
        
        # Her küme için cümleleri birleştir
        chunks = [' '.join(group) for group in grouped_sentences if group]
        
        return chunks
    
    def _cluster_dbscan(self, sentences: List[str], embeddings) -> List[str]:
        """DBSCAN kümeleme ile cümleleri grupla."""
        from sklearn.cluster import DBSCAN
        
        # DBSCAN kümeleme
        clustering = DBSCAN(eps=0.5, min_samples=2)
        clusters = clustering.fit_predict(embeddings)
        
        # Küme sayısını belirle
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        # Kümelere göre cümleleri grupla
        grouped_sentences = [[] for _ in range(n_clusters)]
        outliers = []
        
        cluster_map = {}
        current_idx = 0
        
        for i, cluster_id in enumerate(clusters):
            if cluster_id == -1:
                # Aykırı değerler
                outliers.append(sentences[i])
            else:
                if cluster_id not in cluster_map:
                    cluster_map[cluster_id] = current_idx
                    current_idx += 1
                
                mapped_id = cluster_map[cluster_id]
                if mapped_id >= len(grouped_sentences):
                    # Gerekirse yeni gruplar ekle
                    grouped_sentences.extend([[] for _ in range(mapped_id - len(grouped_sentences) + 1)])
                
                grouped_sentences[mapped_id].append(sentences[i])
        
        # Her küme için cümleleri birleştir
        chunks = [' '.join(group) for group in grouped_sentences if group]
        
        # Aykırı değerleri ekle
        if outliers:
            chunks.append(' '.join(outliers))
        
        return chunks
    
    def _cluster_spectral(self, sentences: List[str], embeddings) -> List[str]:
        """Spektral kümeleme ile cümleleri grupla."""
        from sklearn.cluster import SpectralClustering
        
        # Küme sayısını belirle
        n_clusters = self.n_clusters
        if n_clusters is None:
            # Otomatik küme sayısı belirleme
            n_clusters = max(2, min(len(sentences) // 5, len(sentences) // (self.chunk_size // 100)))
        
        # Spektral kümeleme
        clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
        clusters = clustering.fit_predict(embeddings)
        
        # Kümelere göre cümleleri grupla
        grouped_sentences = [[] for _ in range(n_clusters)]
        for i, cluster_id in enumerate(clusters):
            grouped_sentences[cluster_id].append(sentences[i])
        
        # Her küme için cümleleri birleştir
        chunks = [' '.join(group) for group in grouped_sentences if group]
        
        return chunks
    
    def _cluster_semantic_similarity(self, sentences: List[str], embeddings) -> List[str]:
        """Semantik benzerlik tabanlı gruplama ile cümleleri grupla."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Cümleler arası benzerlik matrisini hesapla
        similarity_matrix = cosine_similarity(embeddings)
        
        # Cümleleri grupla
        current_chunk = [sentences[0]]
        chunks = []
        current_idx = 0
        
        for i in range(1, len(sentences)):
            # Mevcut cümlenin önceki cümleye benzerliği
            similarity = similarity_matrix[current_idx, i]
            
            # Benzerlik eşiğini aşıyorsa aynı gruba ekle
            if similarity >= self.similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                # Yeni bir grup başlat
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_idx = i
        
        # Son grubu ekle
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _optimize_chunk_sizes(self, chunks: List[str]) -> List[str]:
        """Parça boyutlarını optimize et."""
        optimized_chunks = []
        
        for chunk in chunks:
            # Parça çok büyükse böl
            if len(chunk) > self.max_chunk_size:
                # Recursive Character Splitter kullanarak böl
                sub_chunks = self._recursive_split(chunk, self.chunk_size, self.chunk_overlap)
                optimized_chunks.extend(sub_chunks)
            # Parça çok küçükse atla veya birleştir
            elif len(chunk) < self.min_chunk_size:
                # Önceki parça varsa ve birleştirildiğinde max_chunk_size'ı aşmıyorsa birleştir
                if optimized_chunks and len(optimized_chunks[-1]) + len(chunk) <= self.max_chunk_size:
                    optimized_chunks[-1] += " " + chunk
                else:
                    optimized_chunks.append(chunk)
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _recursive_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Metni özyinelemeli olarak böl."""
        # Ayırıcılar
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        # Metin yeterince kısaysa doğrudan döndür
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Parça sonu
            end = start + chunk_size
            
            # Metin sonuna ulaşıldıysa
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Ayırıcı bul
            separator_found = False
            
            for separator in separators:
                if not separator:  # Boş ayırıcı, karakterleri böl
                    chunks.append(text[start:end])
                    separator_found = True
                    break
                
                # Ayırıcı pozisyonunu bul
                separator_pos = text.rfind(separator, start, end)
                
                if separator_pos != -1:
                    # Ayırıcı sonuna kadar al
                    separator_end = separator_pos + len(separator)
                    chunks.append(text[start:separator_end])
                    start = separator_end - chunk_overlap
                    separator_found = True
                    break
            
            # Ayırıcı bulunamadıysa
            if not separator_found:
                chunks.append(text[start:end])
                start = end - chunk_overlap
            
            # Sonsuz döngüyü önle
            if start >= len(text):
                break
        
        return chunks
    
    def chunk_text_with_sliding_window(self, text: str, window_size: int = 100, step_size: int = 50) -> List[Dict[str, str]]:
        """
        Kayan pencere yaklaşımı ile metni böl.
        
        Args:
            text: Bölünecek metin
            window_size: Pencere boyutu (kelime sayısı)
            step_size: Adım boyutu (kelime sayısı)
            
        Returns:
            Parça sözlükleri listesi
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words) - window_size + 1, step_size):
            chunk = ' '.join(words[i:i + window_size])
            chunks.append(chunk)
        
        # Son parçayı ekle
        if len(words) % step_size != 0:
            last_chunk = ' '.join(words[-(len(words) % step_size):])
            chunks.append(last_chunk)
        
        # Parça sözlükleri oluştur
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": i,
                "chunk_text": chunk,
                "source": "sliding_window"
            }
            chunk_dicts.append(chunk_dict)
        
        logger.info(f"Kayan pencere bölümleme tamamlandı. {len(chunk_dicts)} parça oluşturuldu.")
        return chunk_dicts
    
    def chunk_text_with_topic_modeling(self, text: str, num_topics: int = 5) -> List[Dict[str, str]]:
        """
        Konu modelleme ile metni böl.
        
        Args:
            text: Bölünecek metin
            num_topics: Konu sayısı
            
        Returns:
            Parça sözlükleri listesi
        """
        if not self.dependencies["gensim"]:
            logger.warning("gensim yüklü değil. pip install gensim komutuyla yükleyin.")
            return self.chunk_text(text)
        
        try:
            import gensim
            from gensim import corpora
            from gensim.models import LdaModel
            
            # Metni cümlelere böl
            sentences = self._split_into_sentences(text)
            
            # Cümleleri kelimelere böl
            tokenized_sentences = [sentence.lower().split() for sentence in sentences]
            
            # Sözlük oluştur
            dictionary = corpora.Dictionary(tokenized_sentences)
            
            # Belge-terim matrisi oluştur
            corpus = [dictionary.doc2bow(text) for text in tokenized_sentences]
            
            # LDA modeli oluştur
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=10,
                random_state=42
            )
            
            # Her cümle için en olası konuyu belirle
            sentence_topics = []
            for i, bow in enumerate(corpus):
                topics = lda_model.get_document_topics(bow)
                if topics:
                    # En olası konuyu seç
                    most_likely_topic = max(topics, key=lambda x: x[1])[0]
                    sentence_topics.append((i, most_likely_topic))
            
            # Konulara göre cümleleri grupla
            topic_groups = {}
            for i, topic in sentence_topics:
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(sentences[i])
            
            # Her konu için cümleleri birleştir
            chunks = []
            for topic, group in topic_groups.items():
                chunks.append(' '.join(group))
            
            # Parça sözlükleri oluştur
            chunk_dicts = []
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "source": "topic_modeling",
                    "topic_id": i
                }
                chunk_dicts.append(chunk_dict)
            
            logger.info(f"Konu modelleme bölümleme tamamlandı. {len(chunk_dicts)} parça oluşturuldu.")
            return chunk_dicts
            
        except Exception as e:
            logger.error(f"Konu modelleme hatası: {str(e)}")
            return self.chunk_text(text)
    
    def chunk_text_with_textrank(self, text: str) -> List[Dict[str, str]]:
        """
        TextRank algoritması ile metni böl.
        
        Args:
            text: Bölünecek metin
            
        Returns:
            Parça sözlükleri listesi
        """
        if not all([self.dependencies["networkx"], self.dependencies["numpy"], self.dependencies["sentence_transformers"]]):
            logger.warning("networkx, numpy veya sentence_transformers yüklü değil.")
            return self.chunk_text(text)
        
        try:
            import networkx as nx
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Metni cümlelere böl
            sentences = self._split_into_sentences(text)
            
            # Cümle sayısı çok azsa standart bölümleme kullan
            if len(sentences) < 5:
                logger.info("Cümle sayısı çok az, standart bölümleme kullanılıyor")
                return self.chunk_text(text)
            
            # Cümle gömme vektörlerini hesapla
            sentence_embeddings = self.model.encode(sentences, show_progress_bar=False)
            
            # Benzerlik matrisini hesapla
            similarity_matrix = cosine_similarity(sentence_embeddings)
            
            # Benzerlik grafiği oluştur
            graph = nx.from_numpy_array(similarity_matrix)
            
            # TextRank algoritmasını uygula
            scores = nx.pagerank(graph)
            
            # Cümleleri skorlarına göre sırala
            ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
            
            # Önemli cümleleri seç (üst %30)
            num_important = max(3, int(len(sentences) * 0.3))
            important_sentences = [ranked_sentences[i][2] for i in range(num_important)]
            
            # Önemli cümleleri orijinal sırayla düzenle
            important_indices = [ranked_sentences[i][1] for i in range(num_important)]
            important_indices.sort()
            ordered_important_sentences = [sentences[i] for i in important_indices]
            
            # Önemli cümleleri birleştir
            summary = ' '.join(ordered_important_sentences)
            
            # Özeti parçalara böl
            chunks = self._recursive_split(summary, self.chunk_size, self.chunk_overlap)
            
            # Parça sözlükleri oluştur
            chunk_dicts = []
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "source": "textrank"
                }
                chunk_dicts.append(chunk_dict)
            
            logger.info(f"TextRank bölümleme tamamlandı. {len(chunk_dicts)} parça oluşturuldu.")
            return chunk_dicts
            
        except Exception as e:
            logger.error(f"TextRank hatası: {str(e)}")
            return self.chunk_text(text)
    
    def chunk_text_with_hierarchical(self, text: str) -> List[Dict[str, str]]:
        """
        Hiyerarşik bölümleme ile metni böl.
        
        Args:
            text: Bölünecek metin
            
        Returns:
            Parça sözlükleri listesi
        """
        import re
        
        # Başlıkları bul
        heading_pattern = r'^(#+)\s+(.+)$'
        lines = text.split('\n')
        
        # Başlık seviyelerini ve içeriklerini belirle
        sections = []
        current_section = {"level": 0, "title": "", "content": []}
        
        for line in lines:
            match = re.match(heading_pattern, line)
            if match:
                # Yeni bir başlık bulundu
                if current_section["content"]:
                    sections.append(current_section)
                
                level = len(match.group(1))
                title = match.group(2)
                current_section = {"level": level, "title": title, "content": []}
            else:
                # İçerik satırı
                current_section["content"].append(line)
        
        # Son bölümü ekle
        if current_section["content"]:
            sections.append(current_section)
        
        # Bölümleri birleştir
        chunks = []
        for section in sections:
            title = section["title"]
            content = '\n'.join(section["content"])
            chunk = f"{title}\n\n{content}"
            chunks.append(chunk)
        
        # Parça sözlükleri oluştur
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": i,
                "chunk_text": chunk,
                "source": "hierarchical"
            }
            chunk_dicts.append(chunk_dict)
        
        logger.info(f"Hiyerarşik bölümleme tamamlandı. {len(chunk_dicts)} parça oluşturuldu.")
        return chunk_dicts


# Test kodu
if __name__ == "__main__":
    # Bağımlılıkları kontrol et
    dependencies = check_dependencies()
    print("Bağımlılık durumu:")
    for dep, status in dependencies.items():
        print(f"  {dep}: {'Yüklü' if status else 'Yüklü değil'}")
    
    # Test için en az bir NLP kütüphanesi gerekli
    if not dependencies["sentence_transformers"] or not dependencies["sklearn"]:
        print("sentence_transformers ve scikit-learn yüklü değil. Yükleyin:")
        print("  pip install sentence-transformers scikit-learn")
        sys.exit(1)
    
    # Akıllı metin bölücüyü test et
    chunker = SmartTextChunker(
        chunk_size=1000,
        chunk_overlap=100,
        semantic_chunking=True,
        clustering_method="kmeans",
        use_gpu=False,
        language="en"
    )
    
    # Test metni
    test_text = """
    # Artificial Intelligence Overview
    
    Artificial Intelligence (AI) refers to systems that mimic human intelligence and can iteratively 
    improve themselves based on the data they collect. AI encompasses various subfields including 
    machine learning, deep learning, natural language processing, computer vision, and robotics.
    
    ## Applications of AI
    
    AI technologies are revolutionizing many sectors including healthcare, finance, education, 
    transportation, and manufacturing. For example, in healthcare, AI algorithms are being used 
    to diagnose diseases, recommend treatment plans, and accelerate drug discovery.
    
    ## Ethical Considerations
    
    The rapid advancement of AI raises important ethical questions about privacy, bias, 
    accountability, and the future of work. Researchers and policymakers are working to 
    develop frameworks to ensure that AI is developed and deployed in ways that are 
    beneficial, fair, and safe for society.
    
    ## Future Directions
    
    As AI continues to evolve, we can expect to see more sophisticated systems that can handle 
    increasingly complex tasks. Advances in areas such as reinforcement learning, generative models, 
    and multimodal AI are pushing the boundaries of what's possible. However, challenges remain in 
    areas such as common sense reasoning, causal understanding, and creating truly general AI systems.
    """
    
    # Semantik bölümleme
    chunks = chunker.chunk_text(test_text)
    print("\nSemantik Bölümleme:")
    for i, chunk in enumerate(chunks):
        print(f"\nParça {i+1}:")
        print(f"  ID: {chunk['chunk_id']}")
        print(f"  Kaynak: {chunk['source']}")
        print(f"  Metin: {chunk['chunk_text'][:100]}...")
    
    # Kayan pencere bölümleme
    sliding_chunks = chunker.chunk_text_with_sliding_window(test_text, window_size=50, step_size=25)
    print("\nKayan Pencere Bölümleme:")
    print(f"  Toplam parça sayısı: {len(sliding_chunks)}")
    
    # TextRank bölümleme
    if dependencies["networkx"]:
        textrank_chunks = chunker.chunk_text_with_textrank(test_text)
        print("\nTextRank Bölümleme:")
        print(f"  Toplam parça sayısı: {len(textrank_chunks)}")
    
    # Hiyerarşik bölümleme
    hierarchical_chunks = chunker.chunk_text_with_hierarchical(test_text)
    print("\nHiyerarşik Bölümleme:")
    print(f"  Toplam parça sayısı: {len(hierarchical_chunks)}")
    for i, chunk in enumerate(hierarchical_chunks):
        print(f"\nParça {i+1}:")
        print(f"  Metin: {chunk['chunk_text'][:100]}...")
    
    # Konu modelleme bölümleme
    if dependencies["gensim"]:
        topic_chunks = chunker.chunk_text_with_topic_modeling(test_text, num_topics=3)
        print("\nKonu Modelleme Bölümleme:")
        print(f"  Toplam parça sayısı: {len(topic_chunks)}")
