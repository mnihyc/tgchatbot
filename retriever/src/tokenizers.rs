use regex::Regex;
use tantivy::tokenizer::{BoxTokenStream, LowerCaser, SimpleTokenizer, Stemmer, TextAnalyzer, Token, TokenStream, Tokenizer};

#[derive(Clone)]
pub struct MultilingualCaptionTokenizer;

#[derive(Clone)]
pub struct MultilingualTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

impl Tokenizer for MultilingualCaptionTokenizer {
    type TokenStream<'a> = BoxTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let mut tokens: Vec<Token> = Vec::new();
        let word_re = Regex::new(r"[\p{Alphabetic}\p{Number}_+'-]+").unwrap();
        for mat in word_re.find_iter(text) {
            tokens.push(Token {
                offset_from: mat.start(),
                offset_to: mat.end(),
                position: tokens.len(),
                text: mat.as_str().to_lowercase(),
                position_length: 1,
            });
        }
        let chars: Vec<(usize, char)> = text.char_indices().filter(|(_, ch)| is_cjk(*ch)).collect();
        for (idx, (offset, ch)) in chars.iter().enumerate() {
            tokens.push(Token {
                offset_from: *offset,
                offset_to: *offset + ch.len_utf8(),
                position: tokens.len(),
                text: ch.to_string(),
                position_length: 1,
            });
            if idx + 1 < chars.len() {
                let next = chars[idx + 1].1;
                tokens.push(Token {
                    offset_from: *offset,
                    offset_to: chars[idx + 1].0 + next.len_utf8(),
                    position: tokens.len(),
                    text: format!("{}{}", ch, next),
                    position_length: 1,
                });
            }
        }
        BoxTokenStream::new(MultilingualTokenStream { tokens, index: 0 })
    }
}

impl TokenStream for MultilingualTokenStream {
    fn advance(&mut self) -> bool {
        if self.index >= self.tokens.len() {
            return false;
        }
        self.index += 1;
        true
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xac00..=0xd7af | 0xf900..=0xfaff
    )
}

pub fn semantic_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(Stemmer::default())
        .build()
}

pub fn caption_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(MultilingualCaptionTokenizer)
        .filter(LowerCaser)
        .build()
}
