use crate::schema::build_schema;
use crate::tokenizers::{caption_analyzer, semantic_analyzer};
use anyhow::Result;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tantivy::{doc, Index};

pub fn build_index(docs_jsonl: &Path, index_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(index_dir)?;
    let sticker_schema = build_schema();
    let index = Index::create_in_dir(index_dir, sticker_schema.schema.clone())?;
    index.tokenizers().register("caption", caption_analyzer());
    index.tokenizers().register("semantic", semantic_analyzer());
    let mut writer = index.writer(50_000_000)?;
    let reader = BufReader::new(File::open(docs_jsonl)?);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)?;
        writer.add_document(doc!(
            sticker_schema.sticker_id => v["sticker_id"].as_str().unwrap_or_default().to_string(),
            sticker_schema.relative_path => v["relative_path"].as_str().unwrap_or_default().to_string(),
            sticker_schema.source_pack_id => v["source_pack_id"].as_str().unwrap_or_default().to_string(),
            sticker_schema.source_overlay_text_normalized => v["source_overlay_text_normalized"].as_str().unwrap_or_default().to_string(),
            sticker_schema.caption_meaning_en => v["caption_meaning_en"].as_str().unwrap_or_default().to_string(),
            sticker_schema.caption_meaning_zh => v["caption_meaning_zh"].as_str().unwrap_or_default().to_string(),
            sticker_schema.caption_semantic_text => v["caption_semantic_text"].as_str().unwrap_or_default().to_string(),
            sticker_schema.sticker_semantic_text => v["sticker_semantic_text"].as_str().unwrap_or_default().to_string(),
            sticker_schema.style_text => v["style_text"].as_str().unwrap_or_default().to_string(),
            sticker_schema.preview_text => v["preview_text"].as_str().unwrap_or_default().to_string(),
            sticker_schema.caption_dominance_score => v["caption_dominance_score"].as_u64().unwrap_or_default(),
            sticker_schema.source_ocr_confidence_bucket => v["source_ocr_confidence_bucket"].as_u64().unwrap_or_default(),
            sticker_schema.harshness_level => v["harshness_level"].as_u64().unwrap_or_default(),
            sticker_schema.intimacy_level => v["intimacy_level"].as_u64().unwrap_or_default(),
            sticker_schema.meme_dependence_level => v["meme_dependence_level"].as_u64().unwrap_or_default(),
            sticker_schema.style_cluster => v["style_cluster"].as_str().unwrap_or_default().to_string(),
            sticker_schema.style_cluster_id => v["style_cluster_id"].as_u64().unwrap_or_default(),
            sticker_schema.source_pack_hash => v["source_pack_hash"].as_u64().unwrap_or_default(),
            sticker_schema.animated => v["animated"].as_u64().unwrap_or_default(),
            sticker_schema.nsfw_stub_flag => v["nsfw_stub_flag"].as_u64().unwrap_or_default(),
            sticker_schema.selection_notes => v["selection_notes"].as_str().unwrap_or_default().to_string(),
            sticker_schema.raw_json => v.clone(),
        ))?;
    }
    writer.commit()?;
    Ok(())
}
