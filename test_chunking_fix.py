
import re
from typing import List

def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using robust regex.
    Handles common abbreviations and edge cases.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Common abbreviations that shouldn't trigger sentence breaks
    abbreviations = ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'i.e', 'e.g']
    
    # Temporarily replace abbreviations with placeholders
    placeholders = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f'__ABBR{i}__'
        # Match abbreviation followed by period
        pattern = re.escape(abbr) + r'\.'
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
        placeholders[placeholder] = abbr + '.'
    
    # Now split on sentence boundaries: .!? followed by space and capital letter
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    
    # Restore abbreviations
    restored_sentences = []
    for sent in sentences:
        for placeholder, original in placeholders.items():
            sent = sent.replace(placeholder, original)
        restored_sentences.append(sent)
    
    # Clean up and filter empty sentences
    sentences = [s.strip() for s in restored_sentences if s.strip()]
    
    return sentences

# Test cases
test_text = "Dr. Smith went to the store. He bought apples, e.g. red ones. Mrs. Jones was there too."
sentences = _split_into_sentences(test_text)

print(f"Original text: {test_text}")
print(f"Sentences found: {len(sentences)}")
for i, s in enumerate(sentences):
    print(f"{i+1}: {s}")

expected = [
    "Dr. Smith went to the store.",
    "He bought apples, e.g. red ones.",
    "Mrs. Jones was there too."
]

if len(sentences) == 3 and sentences == expected:
    print("\n✅ TEST PASSED: Sentences split correctly handling abbreviations.")
else:
    print("\n❌ TEST FAILED: Output does not match expected.")

