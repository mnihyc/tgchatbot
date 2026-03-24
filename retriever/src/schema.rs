use tantivy::schema::{
    Field, IndexRecordOption, JsonObjectOptions, Schema, SchemaBuilder, TextFieldIndexing, TextOptions,
    FAST, INDEXED, STORED, STRING,
};

#[derive(Clone)]
pub struct StickerSchema {
    pub schema: Schema,
    pub sticker_id: Field,
    pub relative_path: Field,
    pub source_pack_id: Field,
    pub source_overlay_text_normalized: Field,
    pub caption_meaning_en: Field,
    pub caption_meaning_zh: Field,
    pub caption_semantic_text: Field,
    pub sticker_semantic_text: Field,
    pub style_text: Field,
    pub preview_text: Field,
    pub caption_dominance_score: Field,
    pub source_ocr_confidence_bucket: Field,
    pub harshness_level: Field,
    pub intimacy_level: Field,
    pub meme_dependence_level: Field,
    pub style_cluster: Field,
    pub style_cluster_id: Field,
    pub source_pack_hash: Field,
    pub animated: Field,
    pub nsfw_stub_flag: Field,
    pub selection_notes: Field,
    pub raw_json: Field,
}

fn caption_text_options() -> TextOptions {
    TextOptions::default()
        .set_stored()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("caption")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
}

fn semantic_text_options() -> TextOptions {
    TextOptions::default()
        .set_stored()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("semantic")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
}

pub fn build_schema() -> StickerSchema {
    let mut builder = SchemaBuilder::default();
    let sticker_id = builder.add_text_field("sticker_id", STRING | STORED);
    let relative_path = builder.add_text_field("relative_path", STRING | STORED);
    let source_pack_id = builder.add_text_field("source_pack_id", STRING | STORED);
    let source_overlay_text_normalized = builder.add_text_field("source_overlay_text_normalized", caption_text_options());
    let caption_meaning_en = builder.add_text_field("caption_meaning_en", semantic_text_options());
    let caption_meaning_zh = builder.add_text_field("caption_meaning_zh", caption_text_options());
    let caption_semantic_text = builder.add_text_field("caption_semantic_text", semantic_text_options());
    let sticker_semantic_text = builder.add_text_field("sticker_semantic_text", semantic_text_options());
    let style_text = builder.add_text_field("style_text", semantic_text_options());
    let preview_text = builder.add_text_field("preview_text", semantic_text_options());
    let caption_dominance_score = builder.add_u64_field("caption_dominance_score", FAST | INDEXED | STORED);
    let source_ocr_confidence_bucket = builder.add_u64_field("source_ocr_confidence_bucket", FAST | INDEXED | STORED);
    let harshness_level = builder.add_u64_field("harshness_level", FAST | INDEXED | STORED);
    let intimacy_level = builder.add_u64_field("intimacy_level", FAST | INDEXED | STORED);
    let meme_dependence_level = builder.add_u64_field("meme_dependence_level", FAST | INDEXED | STORED);
    let style_cluster = builder.add_text_field("style_cluster", STRING | STORED);
    let style_cluster_id = builder.add_u64_field("style_cluster_id", FAST | INDEXED | STORED);
    let source_pack_hash = builder.add_u64_field("source_pack_hash", FAST | INDEXED | STORED);
    let animated = builder.add_u64_field("animated", FAST | INDEXED | STORED);
    let nsfw_stub_flag = builder.add_u64_field("nsfw_stub_flag", FAST | INDEXED | STORED);
    let selection_notes = builder.add_text_field("selection_notes", semantic_text_options());
    let raw_json = builder.add_json_field("raw_json", JsonObjectOptions::default().set_stored());
    let schema = builder.build();
    StickerSchema {
        schema,
        sticker_id,
        relative_path,
        source_pack_id,
        source_overlay_text_normalized,
        caption_meaning_en,
        caption_meaning_zh,
        caption_semantic_text,
        sticker_semantic_text,
        style_text,
        preview_text,
        caption_dominance_score,
        source_ocr_confidence_bucket,
        harshness_level,
        intimacy_level,
        meme_dependence_level,
        style_cluster,
        style_cluster_id,
        source_pack_hash,
        animated,
        nsfw_stub_flag,
        selection_notes,
        raw_json,
    }
}
