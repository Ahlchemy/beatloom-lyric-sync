"""
English G2P module for SOFA
Uses g2p_en library to convert English words to ARPAbet phonemes
"""
from modules.g2p.base_g2p import BaseG2P


class EnglishG2P(BaseG2P):
    """
    English grapheme-to-phoneme converter using g2p_en.

    Converts English words to ARPAbet phonemes for singing voice alignment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from g2p_en import G2p
            self.g2p = G2p()
        except ImportError:
            raise ImportError(
                "g2p_en is required for English G2P. "
                "Install it with: pip install g2p-en"
            )

    def _g2p(self, input_text):
        """
        Convert English text to phoneme sequence.

        Args:
            input_text: Space-separated English words

        Returns:
            Tuple of (ph_seq, word_seq, ph_idx_to_word_idx)
        """
        # Split input into words
        words = input_text.strip().split()

        # Initialize output sequences
        ph_seq = ["SP"]  # Start with silence
        word_seq = []
        ph_idx_to_word_idx = [-1]  # -1 indicates silence

        for word_idx, word in enumerate(words):
            # Skip empty words
            if not word.strip():
                continue

            # Add word to word sequence
            word_seq.append(word)

            # Convert word to phonemes using g2p_en
            phonemes = self.g2p(word)

            # Filter out non-phoneme characters and convert to uppercase
            # g2p_en returns ARPAbet phonemes
            filtered_phonemes = []
            for ph in phonemes:
                # Remove stress markers (0, 1, 2) and keep only valid phonemes
                ph_clean = ph.rstrip('012')
                if ph_clean and ph_clean not in [' ', "'", '-']:
                    filtered_phonemes.append(ph_clean.upper())

            # Add phonemes to sequence
            for ph in filtered_phonemes:
                ph_seq.append(ph)
                ph_idx_to_word_idx.append(word_idx)

            # Add silence between words
            ph_seq.append("SP")
            ph_idx_to_word_idx.append(-1)

        return ph_seq, word_seq, ph_idx_to_word_idx


if __name__ == "__main__":
    # Test the English G2P
    g2p = EnglishG2P()

    test_text = "Hello world this is a test"
    ph_seq, word_seq, ph_idx_to_word_idx = g2p(test_text)

    print(f"Input: {test_text}")
    print(f"Phonemes: {ph_seq}")
    print(f"Words: {word_seq}")
    print(f"Mapping: {ph_idx_to_word_idx}")
