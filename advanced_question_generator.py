"""
CorpusCrafter: Gelişmiş Soru Üretimi Modülü
-----------------------------------------
Bu modül, PDF'ten CSV veri seti oluşturucu için gelişmiş soru üretimi işlevselliği sağlar.
Bloom taksonomisine dayalı farklı soru türleri ve zorluk seviyelerini destekler.
"""

import os
import time
import logging
import json
import random
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Mevcut modülü içe aktar
import sys
sys.path.append('/home/ubuntu')
from fixed_pdf_to_csv_core import (
    QuestionGenerator, 
    logger, 
    client, 
    TEST_MODE, 
    DEFAULT_MODEL,
    AVAILABLE_MODELS
)

class AdvancedQuestionGenerator(QuestionGenerator):
    """Gelişmiş soru üretimi için genişletilmiş sınıf."""
    
    def __init__(self, 
                 model: str = DEFAULT_MODEL, 
                 system_prompt: Optional[str] = None,
                 question_types: List[str] = None,
                 difficulty_level: str = "medium",
                 cognitive_level: str = "understanding",
                 max_questions_per_chunk: int = 1,
                 include_answer: bool = False,
                 domain_specific: Optional[str] = None):
        """
        Gelişmiş soru oluşturucuyu başlat.
        
        Args:
            model: Kullanılacak AI modeli
            system_prompt: Özel sistem mesajı
            question_types: Oluşturulacak soru türleri listesi
                            ["open_ended", "multiple_choice", "true_false", "fill_in_blank"]
            difficulty_level: Soruların zorluk seviyesi
                              ["easy", "medium", "hard", "expert"]
            cognitive_level: Bloom taksonomisine göre bilişsel seviye
                             ["remembering", "understanding", "applying", 
                              "analyzing", "evaluating", "creating"]
            max_questions_per_chunk: Metin parçası başına maksimum soru sayısı
            include_answer: Sorularla birlikte cevapları da oluştur
            domain_specific: Alan-spesifik sorular için alan adı
                             (örn. "medical", "legal", "technical", "academic")
        """
        super().__init__(model, system_prompt)
        self.question_types = question_types or ["open_ended"]
        self.difficulty_level = difficulty_level
        self.cognitive_level = cognitive_level
        self.max_questions_per_chunk = max_questions_per_chunk
        self.include_answer = include_answer
        self.domain_specific = domain_specific
        
        # Soru türleri için şablonlar
        self._initialize_templates()
        
    def _initialize_templates(self):
        """Farklı soru türleri ve diller için şablonları başlat."""
        self.templates = {
            "en": {
                "open_ended": {
                    "instruction": "Create {difficulty} level open-ended questions that require {cognitive} skills about the following text:",
                    "format": "Question: [Question text]"
                },
                "multiple_choice": {
                    "instruction": "Create {difficulty} level multiple-choice questions that require {cognitive} skills about the following text:",
                    "format": "Question: [Question text]\nOptions:\nA) [Option A]\nB) [Option B]\nC) [Option C]\nD) [Option D]\nCorrect Answer: [Letter]"
                },
                "true_false": {
                    "instruction": "Create {difficulty} level true/false questions that require {cognitive} skills about the following text:",
                    "format": "Question: [Statement]\nAnswer: [True/False]"
                },
                "fill_in_blank": {
                    "instruction": "Create {difficulty} level fill-in-the-blank questions that require {cognitive} skills about the following text:",
                    "format": "Question: [Sentence with _____ for blank]\nAnswer: [Word or phrase]"
                }
            },
            "tr": {
                "open_ended": {
                    "instruction": "Aşağıdaki metin hakkında {difficulty} seviyesinde, {cognitive} becerileri gerektiren açık uçlu sorular oluştur:",
                    "format": "Soru: [Soru metni]"
                },
                "multiple_choice": {
                    "instruction": "Aşağıdaki metin hakkında {difficulty} seviyesinde, {cognitive} becerileri gerektiren çoktan seçmeli sorular oluştur:",
                    "format": "Soru: [Soru metni]\nSeçenekler:\nA) [Seçenek A]\nB) [Seçenek B]\nC) [Seçenek C]\nD) [Seçenek D]\nDoğru Cevap: [Harf]"
                },
                "true_false": {
                    "instruction": "Aşağıdaki metin hakkında {difficulty} seviyesinde, {cognitive} becerileri gerektiren doğru/yanlış soruları oluştur:",
                    "format": "Soru: [İfade]\nCevap: [Doğru/Yanlış]"
                },
                "fill_in_blank": {
                    "instruction": "Aşağıdaki metin hakkında {difficulty} seviyesinde, {cognitive} becerileri gerektiren boşluk doldurma soruları oluştur:",
                    "format": "Soru: [Boşluk için _____ içeren cümle]\nCevap: [Kelime veya ifade]"
                }
            }
        }
        
        # Zorluk seviyesi çevirileri
        self.difficulty_translations = {
            "en": {
                "easy": "easy", 
                "medium": "medium", 
                "hard": "hard", 
                "expert": "expert"
            },
            "tr": {
                "easy": "kolay", 
                "medium": "orta", 
                "hard": "zor", 
                "expert": "uzman"
            }
        }
        
        # Bilişsel seviye çevirileri
        self.cognitive_translations = {
            "en": {
                "remembering": "remembering",
                "understanding": "understanding",
                "applying": "applying",
                "analyzing": "analyzing",
                "evaluating": "evaluating",
                "creating": "creating"
            },
            "tr": {
                "remembering": "hatırlama",
                "understanding": "anlama",
                "applying": "uygulama",
                "analyzing": "analiz etme",
                "evaluating": "değerlendirme",
                "creating": "yaratma"
            }
        }
        
        # Alan-spesifik yönergeler
        self.domain_instructions = {
            "en": {
                "medical": "Focus on medical concepts, terminology, and clinical implications.",
                "legal": "Focus on legal principles, case analysis, and regulatory implications.",
                "technical": "Focus on technical specifications, processes, and implementation details.",
                "academic": "Focus on theoretical frameworks, research methodologies, and scholarly implications."
            },
            "tr": {
                "medical": "Tıbbi kavramlara, terminolojiye ve klinik etkilere odaklan.",
                "legal": "Hukuki ilkelere, dava analizine ve düzenleyici etkilere odaklan.",
                "technical": "Teknik özelliklere, süreçlere ve uygulama detaylarına odaklan.",
                "academic": "Teorik çerçevelere, araştırma metodolojilerine ve akademik etkilere odaklan."
            }
        }
    
    def generate_questions(self, 
                          chunk_text: str, 
                          language: str = "en",
                          retries: int = 3, 
                          backoff_factor: float = 2.0) -> List[Dict[str, str]]:
        """
        Bir metin parçası için birden fazla soru oluştur.
        
        Args:
            chunk_text: Soru oluşturulacak metin parçası
            language: Metnin dili
            retries: Yeniden deneme sayısı
            backoff_factor: Yeniden denemeler için geri çekilme faktörü
            
        Returns:
            Oluşturulan soruların listesi (soru türü, soru metni, cevap)
        """
        # Dil kontrolü
        if language not in ["en", "tr"]:
            logger.warning(f"Dil {language} tam olarak desteklenmiyor. İngilizce kullanılıyor.")
            language = "en"
        
        # Prompt oluştur
        prompt = self._create_prompt(chunk_text, language)
        
        # Sistem mesajını ayarla
        system_message = self._create_system_message(language)
        
        # Test modunda sahte yanıt döndür
        if TEST_MODE:
            return self._generate_mock_questions(language)
        
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
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_params["temperature"],
                    top_p=self.model_params["top_p"],
                    max_tokens=self.model_params["max_tokens"] * 3  # Daha uzun yanıt için
                )
                
                # Yanıtı işle
                raw_response = response.choices[0].message.content.strip()
                questions = self._parse_questions(raw_response, language)
                
                return questions
                
            except Exception as e:
                error_msg = f"API hatası (deneme {attempt+1}/{retries}): {str(e)}"
                logger.warning(error_msg)
                
                if attempt < retries - 1:
                    continue
                else:
                    logger.error(f"Maksimum yeniden deneme sayısına ulaşıldı. Son hata: {str(e)}")
                    # Hata durumunda boş liste döndür
                    return []
        
        # Bu noktaya ulaşılmamalı, ancak güvenlik için
        return []
    
    def _create_prompt(self, chunk_text: str, language: str) -> str:
        """Soru oluşturma için prompt oluştur."""
        # Dile göre zorluk ve bilişsel seviye çevirilerini al
        difficulty = self.difficulty_translations[language][self.difficulty_level]
        cognitive = self.cognitive_translations[language][self.cognitive_level]
        
        # Prompt parçalarını topla
        prompt_parts = []
        
        # Her soru türü için yönerge ekle
        for q_type in self.question_types:
            if q_type in self.templates[language]:
                instruction = self.templates[language][q_type]["instruction"]
                format_example = self.templates[language][q_type]["format"]
                
                # Zorluk ve bilişsel seviyeyi yerleştir
                instruction = instruction.format(difficulty=difficulty, cognitive=cognitive)
                
                prompt_parts.append(f"{instruction}\n{format_example}")
        
        # Alan-spesifik yönerge ekle
        if self.domain_specific and self.domain_specific in self.domain_instructions[language]:
            domain_instruction = self.domain_instructions[language][self.domain_specific]
            prompt_parts.append(domain_instruction)
        
        # Maksimum soru sayısını belirt
        if language == "tr":
            prompt_parts.append(f"Lütfen en fazla {self.max_questions_per_chunk} soru oluştur.")
            if self.include_answer:
                prompt_parts.append("Her soru için bir cevap da ekle.")
        else:
            prompt_parts.append(f"Please create at most {self.max_questions_per_chunk} questions.")
            if self.include_answer:
                prompt_parts.append("Include an answer for each question.")
        
        # Tüm yönergeleri birleştir
        instructions = "\n\n".join(prompt_parts)
        
        # Ana prompt'u oluştur
        if language == "tr":
            prompt = f"""
            {instructions}
            
            Metin:
            {chunk_text}
            
            Sorular:
            """
        else:
            prompt = f"""
            {instructions}
            
            Text:
            {chunk_text}
            
            Questions:
            """
        
        return prompt
    
    def _create_system_message(self, language: str) -> str:
        """Sistem mesajını oluştur."""
        if self.system_prompt:
            return self.system_prompt
        
        if language == "tr":
            return """Sen bir eğitim uzmanısın. Verilen metinler hakkında belirtilen formatta, 
            zorluk seviyesinde ve bilişsel düzeyde sorular oluşturursun. Sorular, metindeki 
            bilgilere dayanmalı ve öğrencilerin anlayışını test etmelidir."""
        else:
            return """You are an educational expert. You create questions about given texts 
            in the specified format, difficulty level, and cognitive level. Questions should 
            be based on the information in the text and test students' understanding."""
    
    def _parse_questions(self, raw_response: str, language: str) -> List[Dict[str, str]]:
        """AI yanıtını ayrıştır ve yapılandırılmış soru listesi oluştur."""
        questions = []
        
        # Farklı soru türleri için ayrıştırma kalıpları
        patterns = {
            "open_ended": {
                "en": r"(?:Question|Q):\s*(.*?)(?:\n|$)",
                "tr": r"(?:Soru|S):\s*(.*?)(?:\n|$)"
            },
            "multiple_choice": {
                "en": r"(?:Question|Q):\s*(.*?)(?:\n|$).*?Options:\s*\n(?:A\)\s*(.*?)(?:\n|$))(?:B\)\s*(.*?)(?:\n|$))(?:C\)\s*(.*?)(?:\n|$))(?:D\)\s*(.*?)(?:\n|$)).*?Correct Answer:\s*([A-D])",
                "tr": r"(?:Soru|S):\s*(.*?)(?:\n|$).*?(?:Seçenekler|Şıklar):\s*\n(?:A\)\s*(.*?)(?:\n|$))(?:B\)\s*(.*?)(?:\n|$))(?:C\)\s*(.*?)(?:\n|$))(?:D\)\s*(.*?)(?:\n|$)).*?(?:Doğru Cevap|Doğru Yanıt):\s*([A-D])"
            },
            "true_false": {
                "en": r"(?:Question|Q):\s*(.*?)(?:\n|$).*?Answer:\s*(True|False)",
                "tr": r"(?:Soru|S):\s*(.*?)(?:\n|$).*?(?:Cevap|Yanıt):\s*(Doğru|Yanlış)"
            },
            "fill_in_blank": {
                "en": r"(?:Question|Q):\s*(.*?)(?:\n|$).*?Answer:\s*(.*?)(?:\n|$)",
                "tr": r"(?:Soru|S):\s*(.*?)(?:\n|$).*?(?:Cevap|Yanıt):\s*(.*?)(?:\n|$)"
            }
        }
        
        # Her soru türü için yanıtı ayrıştır
        for q_type in self.question_types:
            if q_type in patterns:
                pattern = patterns[q_type][language]
                
                if q_type == "multiple_choice":
                    matches = re.finditer(pattern, raw_response, re.DOTALL)
                    for match in matches:
                        question = match.group(1).strip()
                        options = {
                            "A": match.group(2).strip(),
                            "B": match.group(3).strip(),
                            "C": match.group(4).strip(),
                            "D": match.group(5).strip()
                        }
                        correct_answer = match.group(6).strip()
                        
                        questions.append({
                            "type": q_type,
                            "question": question,
                            "options": options,
                            "answer": correct_answer if self.include_answer else None
                        })
                elif q_type == "true_false":
                    matches = re.finditer(pattern, raw_response, re.DOTALL)
                    for match in matches:
                        question = match.group(1).strip()
                        answer = match.group(2).strip()
                        
                        questions.append({
                            "type": q_type,
                            "question": question,
                            "answer": answer if self.include_answer else None
                        })
                elif q_type == "fill_in_blank":
                    matches = re.finditer(pattern, raw_response, re.DOTALL)
                    for match in matches:
                        question = match.group(1).strip()
                        answer = match.group(2).strip()
                        
                        questions.append({
                            "type": q_type,
                            "question": question,
                            "answer": answer if self.include_answer else None
                        })
                else:  # open_ended
                    matches = re.finditer(pattern, raw_response, re.DOTALL)
                    for match in matches:
                        question = match.group(1).strip()
                        
                        # Cevap varsa ayrıştır
                        answer = None
                        if self.include_answer:
                            answer_pattern = {
                                "en": r"(?:Answer|A):\s*(.*?)(?:\n\n|$)",
                                "tr": r"(?:Cevap|Yanıt|C):\s*(.*?)(?:\n\n|$)"
                            }
                            answer_match = re.search(answer_pattern[language], raw_response, re.DOTALL)
                            if answer_match:
                                answer = answer_match.group(1).strip()
                        
                        questions.append({
                            "type": q_type,
                            "question": question,
                            "answer": answer
                        })
        
        # Maksimum soru sayısını kontrol et
        if len(questions) > self.max_questions_per_chunk:
            questions = questions[:self.max_questions_per_chunk]
        
        return questions
    
    def _generate_mock_questions(self, language: str) -> List[Dict[str, str]]:
        """Test modu için sahte sorular oluştur."""
        mock_questions = []
        
        for q_type in self.question_types:
            if q_type == "open_ended":
                if language == "tr":
                    mock_questions.append({
                        "type": q_type,
                        "question": "Bu metindeki ana fikir nedir?",
                        "answer": "Metnin ana fikri yapay zeka teknolojilerinin gelişimidir." if self.include_answer else None
                    })
                else:
                    mock_questions.append({
                        "type": q_type,
                        "question": "What is the main idea of this text?",
                        "answer": "The main idea of the text is the development of AI technologies." if self.include_answer else None
                    })
            elif q_type == "multiple_choice":
                if language == "tr":
                    mock_questions.append({
                        "type": q_type,
                        "question": "Metne göre aşağıdakilerden hangisi doğrudur?",
                        "options": {
                            "A": "Birinci seçenek",
                            "B": "İkinci seçenek",
                            "C": "Üçüncü seçenek",
                            "D": "Dördüncü seçenek"
                        },
                        "answer": "C" if self.include_answer else None
                    })
                else:
                    mock_questions.append({
                        "type": q_type,
                        "question": "According to the text, which of the following is true?",
                        "options": {
                            "A": "First option",
                            "B": "Second option",
                            "C": "Third option",
                            "D": "Fourth option"
                        },
                        "answer": "C" if self.include_answer else None
                    })
            elif q_type == "true_false":
                if language == "tr":
                    mock_questions.append({
                        "type": q_type,
                        "question": "Metne göre yapay zeka teknolojileri hızla gelişmektedir.",
                        "answer": "Doğru" if self.include_answer else None
                    })
                else:
                    mock_questions.append({
                        "type": q_type,
                        "question": "According to the text, AI technologies are developing rapidly.",
                        "answer": "True" if self.include_answer else None
                    })
            elif q_type == "fill_in_blank":
                if language == "tr":
                    mock_questions.append({
                        "type": q_type,
                        "question": "Yapay zeka teknolojileri _____ alanında önemli gelişmeler sağlamıştır.",
                        "answer": "sağlık" if self.include_answer else None
                    })
                else:
                    mock_questions.append({
                        "type": q_type,
                        "question": "AI technologies have made significant advancements in the field of _____.",
                        "answer": "healthcare" if self.include_answer else None
                    })
        
        # Maksimum soru sayısını kontrol et
        if len(mock_questions) > self.max_questions_per_chunk:
            mock_questions = mock_questions[:self.max_questions_per_chunk]
        
        return mock_questions
    
    def generate_question(self, chunk_text: str, language: str = "en",
                        retries: int = 3, backoff_factor: float = 2.0) -> str:
        """
        Uyumluluk için orijinal generate_question metodunu override et.
        
        Args:
            chunk_text: Soru oluşturulacak metin parçası
            language: Metnin dili
            retries: Yeniden deneme sayısı
            backoff_factor: Yeniden denemeler için geri çekilme faktörü
            
        Returns:
            Oluşturulan soru
        """
        # Sadece açık uçlu soru oluştur
        original_question_types = self.question_types
        self.question_types = ["open_ended"]
        
        questions = self.generate_questions(chunk_text, language, retries, backoff_factor)
        
        # Orijinal soru türlerini geri yükle
        self.question_types = original_question_types
        
        if questions and len(questions) > 0:
            return questions[0]["question"]
        else:
            if language == "tr":
                return "Soru oluşturulamadı."
            else:
                return "Could not generate a question."


class AdvancedDatasetCreator:
    """Gelişmiş soru oluşturucu kullanarak veri setleri oluşturma sınıfı."""
    
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
    
    def create_advanced_csv_dataset(self, chunks, 
                                  question_generator: AdvancedQuestionGenerator,
                                  pdf_name: str, language: str = "en",
                                  system_prompt: Optional[str] = None,
                                  max_workers: int = 1) -> str:
        """
        Gelişmiş soru oluşturucu kullanarak CSV veri seti oluştur.
        
        Args:
            chunks: Document nesneleri listesi
            question_generator: AdvancedQuestionGenerator örneği
            pdf_name: PDF dosyasının adı
            language: Metnin dili
            system_prompt: Sistem sütunu için mesaj (isteğe bağlı)
            max_workers: Paralel işleme için maksimum iş parçacığı sayısı
            
        Returns:
            Oluşturulan CSV dosyasının yolu
        """
        logger.info(f"Gelişmiş CSV veri seti oluşturma başlatılıyor (Model: {question_generator.model})")
        
        # Veri çerçevesi oluştur
        data = []
        
        # Soru oluşturma fonksiyonu
        def generate_questions_for_chunk(chunk_with_index):
            index, chunk = chunk_with_index
            chunk_id = chunk.metadata["chunk_id"]
            user_text = chunk.page_content
            
            try:
                # Sorular oluştur
                questions = question_generator.generate_questions(user_text, language)
                
                # Her soru için bir satır ekle
                chunk_data = []
                for q in questions:
                    row = {
                        "chunk_id": chunk_id,
                        "system": system_prompt or "",
                        "user": user_text,
                        "question_type": q["type"],
                        "question": q["question"],
                    }
                    
                    # Soru türüne göre ek alanlar ekle
                    if q["type"] == "multiple_choice":
                        row["options"] = json.dumps(q["options"])
                        if q["answer"]:
                            row["answer"] = q["answer"]
                    elif q["answer"]:
                        row["answer"] = q["answer"]
                    
                    chunk_data.append(row)
                
                return chunk_data
                
            except Exception as e:
                logger.error(f"Parça {index} için soru oluşturma hatası: {str(e)}")
                # Hata durumunda boş liste
                return []
        
        # Paralel işleme kullan (max_workers > 1 ise)
        if max_workers > 1 and not TEST_MODE:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # tqdm ile ilerleme çubuğu
                futures = [executor.submit(generate_questions_for_chunk, (i, chunk)) 
                          for i, chunk in enumerate(chunks)]
                
                for future in tqdm(as_completed(futures), total=len(chunks), desc="Sorular oluşturuluyor"):
                    data.extend(future.result())
        else:
            # Seri işleme
            for i, chunk in enumerate(tqdm(chunks, desc="Sorular oluşturuluyor")):
                data.extend(generate_questions_for_chunk((i, chunk)))
        
        # DataFrame oluştur
        import pandas as pd
        df = pd.DataFrame(data)
        
        # CSV dosya adı oluştur
        base_name = os.path.splitext(os.path.basename(pdf_name))[0]
        model_suffix = question_generator.model.replace("-", "_")
        csv_path = os.path.join(self.output_dir, f"{base_name}_{model_suffix}_advanced_dataset.csv")
        
        # CSV olarak kaydet
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Gelişmiş CSV veri seti oluşturma tamamlandı. Dosya: {csv_path}")
        return csv_path


# Test kodu
if __name__ == "__main__":
    # Test modu etkinleştir
    os.environ["CORPUSCRAFTER_TEST_MODE"] = "true"
    
    # Gelişmiş soru oluşturucuyu test et
    advanced_generator = AdvancedQuestionGenerator(
        model="gpt-4o-mini",
        question_types=["open_ended", "multiple_choice", "true_false", "fill_in_blank"],
        difficulty_level="medium",
        cognitive_level="understanding",
        max_questions_per_chunk=3,
        include_answer=True
    )
    
    # Test metni
    test_text = """
    Yapay zeka (YZ), insan zekasını taklit eden ve toplanan verilere göre yinelemeli olarak 
    kendini iyileştirebilen sistemleri ifade eder. YZ, makine öğrenimi, derin öğrenme, doğal 
    dil işleme, bilgisayarlı görü ve robotik gibi çeşitli alt alanları kapsar.
    
    Yapay zeka teknolojileri, sağlık, finans, eğitim, ulaşım ve üretim gibi birçok sektörde 
    devrim yaratmaktadır. Örneğin, sağlık sektöründe, YZ algoritmaları hastalıkları teşhis 
    etmek, tedavi planları önermek ve ilaç keşfini hızlandırmak için kullanılmaktadır.
    """
    
    # Türkçe sorular oluştur
    tr_questions = advanced_generator.generate_questions(test_text, language="tr")
    
    print("Türkçe Sorular:")
    for q in tr_questions:
        print(f"Tür: {q['type']}")
        print(f"Soru: {q['question']}")
        if "options" in q:
            print("Seçenekler:")
            for key, value in q["options"].items():
                print(f"  {key}) {value}")
        if q["answer"]:
            print(f"Cevap: {q['answer']}")
        print()
    
    # İngilizce sorular oluştur
    en_text = """
    Artificial Intelligence (AI) refers to systems that mimic human intelligence and can iteratively 
    improve themselves based on the data they collect. AI encompasses various subfields including 
    machine learning, deep learning, natural language processing, computer vision, and robotics.
    
    AI technologies are revolutionizing many sectors including healthcare, finance, education, 
    transportation, and manufacturing. For example, in healthcare, AI algorithms are being used 
    to diagnose diseases, recommend treatment plans, and accelerate drug discovery.
    """
    
    en_questions = advanced_generator.generate_questions(en_text, language="en")
    
    print("English Questions:")
    for q in en_questions:
        print(f"Type: {q['type']}")
        print(f"Question: {q['question']}")
        if "options" in q:
            print("Options:")
            for key, value in q["options"].items():
                print(f"  {key}) {value}")
        if q["answer"]:
            print(f"Answer: {q['answer']}")
        print()
