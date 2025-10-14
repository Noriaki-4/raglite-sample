from raglite import RAGLiteConfig, Document, insert_documents, rag
from raglite import _rag as rag_module
import litellm
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple


ToolList = Optional[List[dict[str, Any]]]
ToolChoice = Optional[Any]


@dataclass(frozen=True)
class ModelPolicy:
    name: str
    matcher: Callable[[str], bool]
    configure: Callable[[str], None]
    adjust_tools: Callable[[str, ToolList, ToolChoice], Tuple[ToolList, ToolChoice]]


def _lower(model: str) -> str:
    return (model or "").lower()


def _match_prefix(*prefixes: str) -> Callable[[str], bool]:
    prefixes = tuple(p.lower() for p in prefixes)

    def _matcher(model: str) -> bool:
        lower = _lower(model)
        return any(lower.startswith(prefix) for prefix in prefixes)

    return _matcher


def _identity_adjust(_: str, tools: ToolList, tool_choice: ToolChoice) -> Tuple[ToolList, ToolChoice]:
    return tools, tool_choice


def _anthropic_adjust(_: str, tools: ToolList, tool_choice: ToolChoice) -> Tuple[ToolList, ToolChoice]:
    dummy_tool = {
        "name": "noop",
        "description": "Dummy tool to satisfy Anthropic tool requirement.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    def _convert(tool: dict[str, Any]) -> dict[str, Any]:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            return {
                "name": func.get("name", "function_tool"),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
        return tool

    adjusted_tools: ToolList
    if tools:
        adjusted_tools = [_convert(tool) for tool in tools]
    else:
        adjusted_tools = [dummy_tool]

    adjusted_choice = tool_choice if tool_choice is not None else "auto"
    return adjusted_tools, adjusted_choice


def _configure_anthropic(_: str) -> None:
    litellm.modify_params = True


MODEL_POLICIES: List[ModelPolicy] = [
    ModelPolicy(
        name="anthropic",
        matcher=_match_prefix("anthropic.", "claude"),
        configure=_configure_anthropic,
        adjust_tools=_anthropic_adjust,
    ),
    ModelPolicy(
        name="openai",
        matcher=_match_prefix("gpt", "o1", "o3", "text-davinci"),
        configure=lambda _model: None,
        adjust_tools=_identity_adjust,
    ),
    ModelPolicy(
        name="gemini",
        matcher=_match_prefix("gemini", "google"),
        configure=lambda _model: None,
        adjust_tools=_identity_adjust,
    ),
]


def _resolve_model_policy(model: str) -> Optional[ModelPolicy]:
    for policy in MODEL_POLICIES:
        if policy.matcher(model):
            return policy
    return None


_TOOL_PATCH_INSTALLED = False


def _install_tool_policy_hook():
    global _TOOL_PATCH_INSTALLED
    if _TOOL_PATCH_INSTALLED:
        return

    original_get_tools = getattr(rag_module, "_get_tools", None)
    if original_get_tools is None:
        return

    def patched_get_tools(messages, config):
        tools, tool_choice = original_get_tools(messages, config)
        model_name = getattr(config, "llm", "") or ""
        policy = _resolve_model_policy(model_name)
        if policy:
            tools, tool_choice = policy.adjust_tools(model_name, tools, tool_choice)
        return tools, tool_choice

    rag_module._get_tools = patched_get_tools
    _TOOL_PATCH_INSTALLED = True


def _apply_model_policy(model: str) -> str:
    _install_tool_policy_hook()
    policy = _resolve_model_policy(model)
    if policy:
        policy.configure(model)
        return policy.name
    return "generic"

print("=" * 60)
print("Raglite サンプルプログラム (Claude LLM使用)")
print("=" * 60)

# 環境変数から設定を取得
api_key = os.getenv("ANTHROPIC_API_KEY")
db_path = os.getenv("RAGLITE_DB_PATH", "duckdb:////app/data/raglite.duckdb")
llm_model = os.getenv("RAGLITE_LLM", "claude-3-5-haiku-latest")
embedder_model = os.getenv("RAGLITE_EMBEDDER", "text-embedding-3-small")  # ← 修正

policy_name = _apply_model_policy(llm_model)

# API Key確認
if not api_key:
    print("⚠️  警告: ANTHROPIC_API_KEYが設定されていません")
    print("環境変数を設定してください: export ANTHROPIC_API_KEY='your-key'")
    exit(1)

# OpenAI API Key確認（埋め込み用）
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key and embedder_model.startswith("text-embedding"):
    print("⚠️  警告: OPENAI_API_KEYが設定されていません")
    print("環境変数を設定してください: export OPENAI_API_KEY='your-key'")
    exit(1)

print(f"\n✓ データベース: {db_path}")
print(f"✓ 埋め込みモデル: {embedder_model}")
print(f"✓ LLM: {llm_model}")
print(f"✓ 適用ポリシー: {policy_name}")

# RAGLiteの設定
config = RAGLiteConfig(
    db_url=db_path,
    embedder=embedder_model,
    llm=llm_model
)

print("✓ RAGLite設定完了\n")

# サンプルドキュメントを作成
doc_contents = [
    "Ragliteは軽量なRAG（Retrieval-Augmented Generation）ライブラリです。Pythonで書かれており、簡単に使えます。DuckDBまたはPostgreSQLをバックエンドとして使用できます。",
    "Dockerは、アプリケーションをコンテナという単位でパッケージ化し、どこでも同じように実行できる技術です。コンテナは軽量で、ホストOSのカーネルを共有します。",
    "Docker Composeは、複数のDockerコンテナを定義し、一括で管理するためのツールです。YAML形式の設定ファイルで複数サービスを簡単にオーケストレーションできます。",
    "Ubuntuは、Linuxディストリビューションの一つで、サーバーやデスクトップ環境として広く使われています。Debian系のディストリビューションで、使いやすさに定評があります。",
    "DuckDBは高速な分析データベースシステムです。OLAP処理に最適化されており、軽量で組み込み可能です。SQLiteのようにサーバーレスで動作し、列指向ストレージを採用しています。",
]

# Documentオブジェクトのリストを作成
documents = [Document.from_text(content) for content in doc_contents]

print("📝 ドキュメントを挿入中...")
insert_documents(documents, config=config)
print(f"  ✓ {len(documents)}個のドキュメントを挿入しました\n")

print("=" * 60)

# 質問リスト
questions = [
    "Ragliteとは何ですか？どのような特徴がありますか？",
    "Dockerとは何で、なぜ便利なのですか？",
    "DuckDBの主な利点を教えてください。",
]

# 各質問に対してRAGで回答生成
for question in questions:
    print(f"\n💬 質問: {question}")
    print("-" * 60)
    
    # メッセージ履歴を作成
    messages = [{"role": "user", "content": question}]
    
    # 検索されたチャンクを格納
    chunk_spans = []
    
    # RAG実行: 検索 + Claude LLMで回答生成
    print("🤖 Claude回答:")
    stream = rag(
        messages, 
        on_retrieval=lambda x: chunk_spans.extend(x), 
        config=config
    )
    
    # ストリーミング出力
    for update in stream:
        print(update, end="", flush=True)
    
    print("\n")
    print(f"📚 参照したドキュメント数: {len(chunk_spans)}")
    print("-" * 60)

print("\n" + "=" * 60)
print("サンプルプログラム終了")
print("=" * 60)
