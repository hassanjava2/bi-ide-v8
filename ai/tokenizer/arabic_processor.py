"""
Arabic Text Processor - معالج النصوص العربية
==========================================
Handles Arabic-specific text processing:
- Normalization (ت normalization)
- Diacritic removal
- Different letter forms
"""
import re
from typing import List


class ArabicProcessor:
    """
    Processor for Arabic text
    
    Features:
    - Normalize different forms of alef (أ, إ, آ, ا)
    - Remove diacritics (tashkeel)
    - Handle Arabic punctuation
    - Normalize taa marbuta (ة/ه)
    - Normalize yaa (ي/ى)
    """
    
    # Arabic character ranges
    ARABIC_LETTERS = r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    
    # Diacritics (Harakat)
    DIACRITICS = "\u064B-\u065F\u0670\u0640"
    
    # Punctuation
    ARABIC_PUNCTUATION = "\u060C\u061B\u061F\u066A\u066B\u066C"
    
    def __init__(self):
        # Normalization mappings
        self.alef_variants = {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ٱ": "ا"
        }
        
        self.yaa_variants = {
            "ى": "ي"
        }
        
        self.taa_variants = {
            "ة": "ه"
        }
    
    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text
        
        Steps:
        1. Remove diacritics
        2. Normalize alef variants
        3. Normalize taa marbuta
        4. Normalize yaa
        5. Normalize whitespace
        """
        # Remove diacritics
        text = self.remove_diacritics(text)
        
        # Normalize alef variants
        for variant, normalized in self.alef_variants.items():
            text = text.replace(variant, normalized)
        
        # Normalize taa marbuta
        for variant, normalized in self.taa_variants.items():
            text = text.replace(variant, normalized)
        
        # Normalize yaa
        for variant, normalized in self.yaa_variants.items():
            text = text.replace(variant, normalized)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (tashkeel)"""
        return re.sub(f"[{self.DIACRITICS}]", "", text)
    
    def remove_punctuation(self, text: str) -> str:
        """Remove Arabic punctuation"""
        return re.sub(f"[{self.ARABIC_PUNCTUATION}]", " ", text)
    
    def is_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        return bool(re.search(f"[{self.ARABIC_LETTERS}]", text))
    
    def split_arabic_words(self, text: str) -> List[str]:
        """Split text into Arabic words"""
        # Match Arabic words
        arabic_words = re.findall(f"[{self.ARABIC_LETTERS}]+", text)
        return arabic_words
    
    def count_arabic_words(self, text: str) -> int:
        """Count Arabic words in text"""
        return len(self.split_arabic_words(text))
    
    def preprocess_for_tokenization(self, text: str) -> str:
        """
        Full preprocessing pipeline for tokenization
        
        This is the recommended entry point before tokenization
        """
        # Normalize
        text = self.normalize(text)
        
        # Add spaces around punctuation
        text = re.sub(r"([^\w\s])", r" \1 ", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    def get_word_root(self, word: str) -> str:
        """
        Extract root of Arabic word (simplified)
        
        Note: This is a simplified version. For production,
        consider using a proper Arabic stemmer like Farasa or ISRI
        """
        # Remove non-Arabic characters
        word = re.sub(f"[^{self.ARABIC_LETTERS}]", "", word)
        
        # Normalize
        word = self.normalize(word)
        
        # Simple heuristic: remove common prefixes and suffixes
        prefixes = ["ال", "بال", "كال", "فال", "لل", "ولل"]
        suffixes = ["ة", "ات", "ين", "ون", "ان", "وا", "ت", "نا", "تم", "كن"]
        
        # Remove prefixes
        for prefix in prefixes:
            if word.startswith(prefix):
                word = word[len(prefix):]
                break
        
        # Remove suffixes
        for suffix in suffixes:
            if word.endswith(suffix):
                word = word[:-len(suffix)]
                break
        
        return word


# Convenience function
def normalize_arabic(text: str) -> str:
    """Quick Arabic normalization"""
    processor = ArabicProcessor()
    return processor.normalize(text)


if __name__ == "__main__":
    # Test
    processor = ArabicProcessor()
    
    texts = [
        "مرحباً بالعالم!",
        "الذكاء الإصطناعي",
        "مكتبة",
        "القُرآن الكَرِيم"
    ]
    
    for text in texts:
        print(f"Original: {text}")
        print(f"Normalized: {processor.normalize(text)}")
        print(f"Preprocessed: {processor.preprocess_for_tokenization(text)}")
        print()
