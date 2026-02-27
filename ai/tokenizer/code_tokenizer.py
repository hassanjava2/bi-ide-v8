"""
Code Tokenizer - توكنيزر الأكواد
=============================
Specialized tokenization for programming languages
"""
import re
from typing import List, Dict, Optional
from enum import Enum


class ProgrammingLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    RUST = "rust"
    GO = "go"
    SQL = "sql"
    BASH = "bash"
    UNKNOWN = "unknown"


class CodeTokenizer:
    """
    Tokenizer optimized for code
    
    Features:
    - Language detection
    - Preserve indentation
    - Handle strings and comments
    - Special handling for operators
    """
    
    # Language patterns for detection
    LANGUAGE_PATTERNS = {
        ProgrammingLanguage.PYTHON: [
            r"def\s+\w+\s*\(",
            r"class\s+\w+\s*[:\(]",
            r"import\s+\w+",
            r"from\s+\w+\s+import",
            r"if\s+__name__\s*==\s*['\"]__main__['\"]"
        ],
        ProgrammingLanguage.JAVASCRIPT: [
            r"function\s+\w+\s*\(",
            r"const\s+\w+\s*=",
            r"let\s+\w+\s*=",
            r"var\s+\w+\s*=",
            r"=>\s*\{"
        ],
        ProgrammingLanguage.TYPESCRIPT: [
            r"interface\s+\w+",
            r"type\s+\w+\s*=",
            r":\s*(string|number|boolean|any)\s*[;,=)]"
        ],
        ProgrammingLanguage.JAVA: [
            r"public\s+(class|static|void)",
            r"private\s+\w+",
            r"System\.out\.print"
        ],
        ProgrammingLanguage.SQL: [
            r"SELECT\s+.*\s+FROM",
            r"INSERT\s+INTO",
            r"UPDATE\s+\w+\s+SET",
            r"CREATE\s+TABLE"
        ]
    }
    
    # Common code tokens
    OPERATORS = [
        "+", "-", "*", "/", "%", "=",
        "==", "!=", "<", ">", "<=", ">=",
        "&&", "||", "!",
        "&", "|", "^", "~", "<<", ">>",
        "++", "--",
        "+=", "-=", "*=", "/=", "%=",
        "**", "//"
    ]
    
    BRACKETS = ["(", ")", "[", "]", "{", "}", "<", ">"]
    
    PUNCTUATION = [";", ",", ".", ":", "?", "@", "#", "$"]
    
    def __init__(self):
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[ProgrammingLanguage, List[re.Pattern]]:
        """Compile regex patterns for language detection"""
        compiled = {}
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            compiled[lang] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def detect_language(self, code: str) -> ProgrammingLanguage:
        """Detect programming language from code"""
        scores = {lang: 0 for lang in ProgrammingLanguage}
        
        for lang, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(code):
                    scores[lang] += 1
        
        # Return language with highest score
        best_lang = max(scores, key=scores.get)
        
        if scores[best_lang] == 0:
            return ProgrammingLanguage.UNKNOWN
        
        return best_lang
    
    def tokenize(self, code: str, language: Optional[ProgrammingLanguage] = None) -> List[str]:
        """
        Tokenize code into tokens
        
        Preserves:
        - Identifiers (variable/function names)
        - Keywords
        - Operators
        - Literals (strings, numbers)
        - Comments
        """
        if language is None:
            language = self.detect_language(code)
        
        tokens = []
        lines = code.split("\n")
        
        for line in lines:
            line_tokens = self._tokenize_line(line)
            tokens.extend(line_tokens)
            tokens.append("\n")  # Preserve newlines
        
        return tokens
    
    def _tokenize_line(self, line: str) -> List[str]:
        """Tokenize a single line of code"""
        tokens = []
        i = 0
        
        while i < len(line):
            char = line[i]
            
            # Skip whitespace (but track indentation)
            if char == " ":
                # Count leading spaces
                space_count = 0
                while i < len(line) and line[i] == " ":
                    space_count += 1
                    i += 1
                if space_count > 0 and not tokens:
                    # Leading indentation
                    tokens.append(" " * space_count)
                continue
            
            # String literals
            if char in ['"', "'", "`"]:
                string_token, i = self._read_string(line, i)
                tokens.append(string_token)
                continue
            
            # Numbers
            if char.isdigit():
                num_token, i = self._read_number(line, i)
                tokens.append(num_token)
                continue
            
            # Identifiers and keywords
            if char.isalpha() or char == "_":
                ident_token, i = self._read_identifier(line, i)
                tokens.append(ident_token)
                continue
            
            # Operators (check multi-char first)
            op_found = False
            for op in sorted(self.OPERATORS, key=len, reverse=True):
                if line[i:i+len(op)] == op:
                    tokens.append(op)
                    i += len(op)
                    op_found = True
                    break
            
            if op_found:
                continue
            
            # Brackets and punctuation
            if char in self.BRACKETS + self.PUNCTUATION:
                tokens.append(char)
                i += 1
                continue
            
            # Skip unknown characters
            i += 1
        
        return tokens
    
    def _read_string(self, line: str, start: int) -> tuple:
        """Read a string literal"""
        quote = line[start]
        end = start + 1
        
        while end < len(line):
            if line[end] == quote and line[end-1] != "\\":
                end += 1
                break
            end += 1
        
        return line[start:end], end
    
    def _read_number(self, line: str, start: int) -> tuple:
        """Read a number literal"""
        end = start
        has_dot = False
        
        while end < len(line):
            char = line[end]
            if char.isdigit():
                end += 1
            elif char == "." and not has_dot:
                has_dot = True
                end += 1
            elif char in ["e", "E"] and end > start:
                # Scientific notation
                if end + 1 < len(line) and line[end + 1] in ["+", "-"]:
                    end += 2
                else:
                    end += 1
            else:
                break
        
        return line[start:end], end
    
    def _read_identifier(self, line: str, start: int) -> tuple:
        """Read an identifier or keyword"""
        end = start
        
        while end < len(line) and (line[end].isalnum() or line[end] == "_"):
            end += 1
        
        return line[start:end], end
    
    def extract_comments(self, code: str, language: Optional[ProgrammingLanguage] = None) -> List[str]:
        """Extract comments from code"""
        if language is None:
            language = self.detect_language(code)
        
        comments = []
        
        if language in [ProgrammingLanguage.PYTHON, ProgrammingLanguage.BASH]:
            # # style comments
            for line in code.split("\n"):
                if "#" in line:
                    comment = line[line.index("#"):]
                    comments.append(comment)
        
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT, ProgrammingLanguage.JAVA, ProgrammingLanguage.CPP, ProgrammingLanguage.RUST, ProgrammingLanguage.GO]:
            # // and /* */ style comments
            # Single line
            for line in code.split("\n"):
                if "//" in line:
                    comment = line[line.index("//"):]
                    comments.append(comment)
            
            # Multi-line
            multi_line_pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
            comments.extend(multi_line_pattern.findall(code))
        
        elif language == ProgrammingLanguage.SQL:
            # -- style comments
            for line in code.split("\n"):
                if "--" in line:
                    comment = line[line.index("--"):]
                    comments.append(comment)
        
        return comments
    
    def remove_comments(self, code: str, language: Optional[ProgrammingLanguage] = None) -> str:
        """Remove comments from code"""
        if language is None:
            language = self.detect_language(code)
        
        if language in [ProgrammingLanguage.PYTHON, ProgrammingLanguage.BASH]:
            # Remove # comments
            lines = []
            for line in code.split("\n"):
                if "#" in line:
                    line = line[:line.index("#")]
                lines.append(line)
            return "\n".join(lines)
        
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            # Remove // and /* */ comments
            code = re.sub(r"//.*", "", code)
            code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
            return code
        
        return code
    
    def get_code_statistics(self, code: str) -> Dict:
        """Get statistics about code"""
        language = self.detect_language(code)
        tokens = self.tokenize(code, language)
        comments = self.extract_comments(code, language)
        
        # Count token types
        identifiers = [t for t in tokens if t.isidentifier()]
        numbers = [t for t in tokens if t.replace(".", "").isdigit()]
        operators = [t for t in tokens if t in self.OPERATORS]
        
        return {
            "language": language.value,
            "total_lines": len(code.split("\n")),
            "code_lines": len([l for l in code.split("\n") if l.strip()]),
            "comment_lines": len(comments),
            "total_tokens": len(tokens),
            "identifiers": len(identifiers),
            "numbers": len(numbers),
            "operators": len(operators),
            "comment_ratio": len(comments) / len(code.split("\n")) if code else 0
        }


if __name__ == "__main__":
    # Test
    tokenizer = CodeTokenizer()
    
    python_code = '''
def hello_world():
    # This is a comment
    x = 10 + 20
    name = "Python"
    return f"Hello, {name}!"
'''
    
    print("Language:", tokenizer.detect_language(python_code))
    print("Tokens:", tokenizer.tokenize(python_code)[:20])
    print("Stats:", tokenizer.get_code_statistics(python_code))
