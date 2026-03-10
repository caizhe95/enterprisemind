"""文档处理 - 企业级多策略分割"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
import re
import hashlib
import numpy as np


class AdvancedDocumentProcessor:
    """结构感知文档处理器（支持语义分块与层次化分割）"""

    def __init__(self, embedding_model=None, chunk_size: int = 512):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.default_overlap = 50
        self.chunk_profiles = {
            "markdown": {
                "chunk_size": 720,
                "chunk_overlap": 100,
                "semantic_threshold": 0.50,
            },
            "pdf": {"chunk_size": 520, "chunk_overlap": 60, "semantic_threshold": 0.48},
            "word": {
                "chunk_size": 520,
                "chunk_overlap": 60,
                "semantic_threshold": 0.50,
            },
            "text": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "semantic_threshold": 0.50,
            },
            "json": {
                "chunk_size": 420,
                "chunk_overlap": 40,
                "semantic_threshold": 0.55,
            },
            "csv": {"chunk_size": 420, "chunk_overlap": 30, "semantic_threshold": 0.60},
            "html": {
                "chunk_size": 520,
                "chunk_overlap": 60,
                "semantic_threshold": 0.50,
            },
            "unknown": {
                "chunk_size": chunk_size,
                "chunk_overlap": self.default_overlap,
                "semantic_threshold": 0.50,
            },
        }
        self.basic_splitter = self._create_splitter(chunk_size, self.default_overlap)
        self._current_splitter = self.basic_splitter
        self._current_chunk_size = chunk_size
        self._current_semantic_threshold = 0.50

    def _create_splitter(
        self, chunk_size: int, chunk_overlap: int
    ) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_count,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        )

    def _token_count(self, text: str) -> int:
        """更精确的token估算（中文按字，英文按词）"""
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_words = len(re.findall(r"[a-zA-Z]+", text))
        return chinese_chars + english_words

    def _get_source_type(self, doc: Document) -> str:
        source_type = (doc.metadata or {}).get("source_type", "").lower()
        if source_type:
            return source_type
        file_name = (doc.metadata or {}).get("file_name", "")
        ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
        ext_map = {
            "md": "markdown",
            "pdf": "pdf",
            "doc": "word",
            "docx": "word",
            "txt": "text",
            "csv": "csv",
            "json": "json",
            "html": "html",
        }
        return ext_map.get(ext, "unknown")

    def _set_profile_for_doc(self, doc: Document):
        source_type = self._get_source_type(doc)
        profile = self.chunk_profiles.get(source_type, self.chunk_profiles["unknown"])
        self._current_chunk_size = int(profile["chunk_size"])
        self._current_semantic_threshold = float(profile["semantic_threshold"])
        self._current_splitter = self._create_splitter(
            int(profile["chunk_size"]), int(profile["chunk_overlap"])
        )

    def _build_stable_chunk_id(
        self, doc: Document, chunk: Document, chunk_index: int
    ) -> str:
        source_name = (doc.metadata or {}).get(
            "file_name", (doc.metadata or {}).get("source", "doc")
        )
        normalized = re.sub(r"\s+", " ", chunk.page_content.strip())[:1200]
        digest = hashlib.sha1(
            f"{source_name}|{chunk_index}|{normalized}".encode("utf-8")
        ).hexdigest()[:16]
        return f"{source_name}__{digest}"

    def process(
        self, documents: List[Document], strategy: str = "auto"
    ) -> List[Document]:
        """
        处理文档入口
        strategy: auto(自动检测), markdown, semantic, hierarchical, standard
        """
        processed = []

        for doc in documents:
            self._set_profile_for_doc(doc)
            if strategy == "auto":
                actual_strategy = self._detect_strategy(doc)
            else:
                actual_strategy = strategy

            if actual_strategy == "markdown":
                chunks = self._split_markdown_advanced(doc)
            elif actual_strategy == "semantic" and self.embedding_model:
                chunks = self._split_semantic(doc)
            elif actual_strategy == "hierarchical":
                chunks = self._split_hierarchical(doc)
            else:
                chunks = self._split_standard(doc)

            # 添加通用元数据（强制合并原始metadata，修复file_name丢失问题）
            for i, chunk in enumerate(chunks):
                # 关键修复：先合并原始文档的metadata（确保file_name等保留）
                if doc.metadata:
                    chunk.metadata = {**doc.metadata, **chunk.metadata}

                # 生成chunk_id时使用file_name
                source_name = doc.metadata.get(
                    "file_name", doc.metadata.get("source", "doc")
                )
                chunk.metadata.update(
                    {
                        "chunk_id": self._build_stable_chunk_id(doc, chunk, i),
                        "chunk_index": i,
                        "processing_strategy": actual_strategy,
                        "token_count": self._token_count(chunk.page_content),
                        "char_count": len(chunk.page_content),
                        "source_name": source_name,
                    }
                )
                processed.append(chunk)

        return processed

    def _detect_strategy(self, doc: Document) -> str:
        """自动检测文档类型"""
        content = doc.page_content[:1000]

        # Markdown检测
        if re.search(r"^#{1,3}\s+", content, re.MULTILINE):
            return "markdown"
        # 长文本且提供embedding模型，使用语义分块
        elif self.embedding_model and len(content) > 2000:
            return "semantic"
        else:
            return "standard"

    def _split_markdown_advanced(self, doc: Document) -> List[Document]:
        """增强Markdown分割（保护表格和代码块）"""
        content = doc.page_content
        placeholders = {}
        placeholder_id = 0

        # 保护代码块
        def protect_code(match):
            nonlocal placeholder_id
            key = f"__CODE_{placeholder_id}__"
            placeholders[key] = match.group(0)
            placeholder_id += 1
            return key

        # 保护表格
        def protect_table(match):
            nonlocal placeholder_id
            key = f"__TABLE_{placeholder_id}__"
            placeholders[key] = match.group(0)
            placeholder_id += 1
            return key

        content = re.sub(r"```[\s\S]*?```", protect_code, content)
        content = re.sub(r"(\|[^\n]+\|\n)+", protect_table, content)

        # 按标题分割
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        try:
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            chunks = md_splitter.split_text(content)
        except Exception:
            chunks = self._current_splitter.create_documents([content])

        # 恢复保护内容并提取层级路径
        final_chunks = []
        for chunk in chunks:
            chunk_text = chunk.page_content
            for key, val in placeholders.items():
                chunk_text = chunk_text.replace(key, val)
            chunk_text = self._append_table_field_index(chunk_text)

            # 优先使用 MarkdownHeaderTextSplitter 提供的层级字段
            h1 = chunk.metadata.get("H1")
            h2 = chunk.metadata.get("H2")
            h3 = chunk.metadata.get("H3")
            header_path = " > ".join(
                [x for x in [h1, h2, h3] if x]
            ) or self._extract_header_path(doc.page_content, chunk_text)

            # 关键修复：合并原始文档的metadata（保留file_name、source_type等）
            merged_metadata = {**doc.metadata, **chunk.metadata}

            chunk.metadata.update(
                {
                    **merged_metadata,
                    "header_path": header_path,
                    "chunk_type": "markdown_section",
                }
            )
            chunk.page_content = chunk_text
            # 超长块二次切分，避免单块过大影响召回稳定性
            if self._token_count(chunk.page_content) > int(self._current_chunk_size * 1.3):
                sub_chunks = self._current_splitter.create_documents([chunk.page_content])
                for idx, sub in enumerate(sub_chunks, start=1):
                    sub.metadata.update(
                        {
                            **chunk.metadata,
                            "sub_chunk_index": idx,
                            "chunk_type": "markdown_sub_section",
                        }
                    )
                    final_chunks.append(sub)
            else:
                final_chunks.append(chunk)

        return final_chunks if final_chunks else [doc]

    def _append_table_field_index(self, text: str) -> str:
        """
        为 Markdown 表格追加“字段:值”索引，提升字段类查询（如“续航”）召回率。
        不改变原始表格，仅在尾部附加轻量索引文本。
        """
        table_blocks = re.findall(r"(?:^\|.*\|\s*$\n?)+", text, flags=re.MULTILINE)
        if not table_blocks:
            return text

        kv_lines = []
        for block in table_blocks:
            for line in block.splitlines():
                line = line.strip()
                if not line.startswith("|") or line.count("|") < 3:
                    continue
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) < 2:
                    continue

                # 跳过分隔行和表头行
                if all(re.fullmatch(r"[:\- ]+", c or "") for c in cells):
                    continue
                if cells[0] in {"项目", "字段", "参数", "指标", "名称"} and cells[
                    1
                ] in {"参数", "取值", "值", "说明"}:
                    continue

                key, value = cells[0], cells[1]
                if key and value:
                    kv_lines.append(f"{key}: {value}")

        if not kv_lines:
            return text

        deduped = list(dict.fromkeys(kv_lines))[:30]
        return f"{text}\n\n[表格字段索引]\n" + "\n".join(deduped)

    def _extract_header_path(self, full_content: str, chunk_content: str) -> str:
        """提取文档层级路径"""
        start_idx = full_content.find(chunk_content[:50])
        if start_idx == -1:
            return ""

        preceding = full_content[:start_idx]
        h1 = re.findall(r"^#\s+(.+)$", preceding, re.MULTILINE)
        h2 = re.findall(r"^##\s+(.+)$", preceding, re.MULTILINE)

        path = []
        if h1:
            path.append(h1[-1])
        if h2:
            path.append(h2[-1])

        return " > ".join(path)

    def _split_semantic(self, doc: Document) -> List[Document]:
        """语义分块：基于Embedding相似度确定边界"""
        if not self.embedding_model:
            return self._split_standard(doc)

        # 按句子切分
        sentences = re.split(r"(?<=[。！？.!?])\s+", doc.page_content)
        if len(sentences) <= 3:
            return [doc]

        # 计算embedding（批量处理以加速）
        try:
            embeddings = self.embedding_model.embed_documents(sentences)
        except Exception:
            return self._split_standard(doc)

        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])

        for i in range(1, len(sentences)):
            # 计算与前一句的相似度
            prev_emb = np.array(embeddings[i - 1])
            curr_emb = np.array(embeddings[i])
            similarity = np.dot(prev_emb, curr_emb) / (
                np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb) + 1e-8
            )

            sentence_size = len(sentences[i])

            # 切割条件：块够大且语义断裂，或块过大
            if (
                current_size + sentence_size > self._current_chunk_size * 0.8
                and similarity < self._current_semantic_threshold
            ) or (current_size + sentence_size > self._current_chunk_size):
                new_doc = Document(
                    page_content="".join(current_chunk),
                    metadata={
                        **doc.metadata,
                        "chunk_type": "semantic",
                    },  # 保留原始metadata
                )
                chunks.append(new_doc)
                current_chunk = [sentences[i]]
                current_size = sentence_size
            else:
                current_chunk.append(sentences[i])
                current_size += sentence_size

        if current_chunk:
            chunks.append(
                Document(
                    page_content="".join(current_chunk),
                    metadata={**doc.metadata, "chunk_type": "semantic"},
                )
            )

        return chunks

    def _split_hierarchical(self, doc: Document) -> List[Document]:
        """层次化分块：保留父子关系"""
        base_chunks = self._split_standard(doc)
        enriched = []

        for i, chunk in enumerate(base_chunks):
            prev_context = base_chunks[i - 1].page_content[-100:] if i > 0 else ""
            next_context = (
                base_chunks[i + 1].page_content[:100]
                if i < len(base_chunks) - 1
                else ""
            )

            # 保留原始metadata并添加层级信息
            chunk.metadata = {
                **doc.metadata,
                **chunk.metadata,
                "total_chunks": len(base_chunks),
                "prev_summary": prev_context[:50] + "..." if prev_context else "",
                "next_summary": next_context[:50] + "..." if next_context else "",
                "chunk_type": "hierarchical",
            }
            enriched.append(chunk)

        return enriched

    def _split_standard(self, doc: Document) -> List[Document]:
        """标准递归分割"""
        chunks = self._current_splitter.split_documents([doc])
        for chunk in chunks:
            # 保留原始metadata
            chunk.metadata = {
                **doc.metadata,
                **chunk.metadata,
                "chunk_type": "standard",
            }
        return chunks


# 保持向后兼容的别名
DocumentProcessor = AdvancedDocumentProcessor
