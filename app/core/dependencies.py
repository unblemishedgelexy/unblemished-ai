from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.logger import StructuredLogger, setup_logger
from app.core.runtime_analyzer import RuntimeAnalyzer
from app.core.task_manager import TaskManager
from app.core.telemetry_exporter import TelemetryExporter
from app.repositories.goal_repository import PostgresGoalRepository, SQLiteGoalRepository
from app.repositories.memory_repository import PostgresMemoryRepository, SQLiteMemoryRepository
from app.repositories.profile_repository import PostgresProfileRepository, SQLiteProfileRepository
from app.repositories.skill_repository import PostgresSkillRepository, SQLiteSkillRepository
from app.services.behavior.directness_enforcer import DirectnessEnforcer
from app.services.behavior.human_coherence_scorer import HumanCoherenceScorer
from app.services.behavior.response_match_predictor import ResponseMatchPredictor
from app.services.behavior.response_style_engine import ResponseStyleEngine
from app.services.behavior.topic_alignment_checker import TopicAlignmentChecker
from app.services.behavior.vagueness_detector import VaguenessDetector
from app.services.brain.brain_interface import BrainInterface
from app.services.brain.evaluation_engine import EvaluationEngine
from app.services.brain.execution_controller import ExecutionController
from app.services.brain.fallback_engine import FallbackEngine
from app.services.brain.model_adapter import ModelAdapter
from app.services.brain.orchestrator import BrainOrchestrator
from app.services.brain.planning_engine import PlanningEngine
from app.services.brain.prompt_builder import PromptBuilder
from app.services.brain.response_post_processor import ResponsePostProcessor
from app.services.brain.reflection_engine import ReflectionEngine
from app.services.brain.safety_classifier import SafetyClassifier
from app.services.brain.safe_web_lookup import SafeWebLookup
from app.services.brain.telemetry_engine import TelemetryEngine
from app.services.embeddings.embedding_adapter import EmbeddingAdapter
from app.services.embeddings.embedding_interface import EmbeddingInterface
from app.services.formatter.response_formatter import ResponseFormatter
from app.services.goals.goal_interface import GoalInterface
from app.services.goals.goal_store import GoalStore
from app.services.intent.intent_analyzer import IntentAnalyzer
from app.services.memory.memory_interface import MemoryInterface
from app.services.memory.memory_retriever import MemoryRetriever
from app.services.memory.memory_store import MemoryStore
from app.services.memory.memory_summarizer import MemorySummarizer
from app.services.memory.policies.memory_decay_policy import MemoryDecayPolicy
from app.services.memory.policies.memory_importance_policy import MemoryImportancePolicy
from app.services.memory.policies.memory_ranking_policy import MemoryRankingPolicy
from app.services.personality.emotional_continuity_engine import EmotionalContinuityEngine
from app.services.personality.identity_core import IdentityCore
from app.services.personality.personality_profile import PersonalityProfileService
from app.services.personality.safety_guardrails import SafetyGuardrails
from app.services.profile.user_profile_interface import UserProfileInterface
from app.services.profile.user_profile_store import UserProfileStore
from app.services.relationship.relationship_memory_store import RelationshipMemoryStore
from app.services.relationship.relationship_state_engine import RelationshipStateEngine
from app.services.router.model_router import ModelRouter
from app.services.skills.skill_interface import SkillInterface
from app.services.tools.tool_executor import ToolExecutor
from app.services.tools.tool_interface import ToolInterface
from app.services.tools.tool_registry import ToolRegistry


@dataclass(slots=True)
class ServiceContainer:
    settings: Settings
    logger: StructuredLogger
    task_manager: TaskManager
    brain: BrainInterface
    embedding_interface: EmbeddingInterface
    memory_interface: MemoryInterface
    user_profile_interface: UserProfileInterface
    goal_interface: GoalInterface
    tool_interface: ToolInterface
    model_router: ModelRouter
    memory_store: MemoryStore
    profile_store: UserProfileStore
    goal_store: GoalStore
    relationship_memory_store: RelationshipMemoryStore
    relationship_state_engine: RelationshipStateEngine
    emotional_continuity_engine: EmotionalContinuityEngine
    safety_guardrails: SafetyGuardrails
    skill_interface: SkillInterface
    telemetry_exporter: TelemetryExporter
    runtime_analyzer: RuntimeAnalyzer
    response_style_engine: ResponseStyleEngine
    response_match_predictor: ResponseMatchPredictor
    directness_enforcer: DirectnessEnforcer
    vagueness_detector: VaguenessDetector
    topic_alignment_checker: TopicAlignmentChecker
    human_coherence_scorer: HumanCoherenceScorer


def create_container(
    settings: Settings | None = None,
    model_adapter: ModelAdapter | None = None,
) -> ServiceContainer:
    runtime_settings = settings or get_settings()
    logger = StructuredLogger(setup_logger(level=runtime_settings.log_level))
    task_manager = TaskManager(logger=logger)
    telemetry_exporter = TelemetryExporter(
        logger=logger,
        enabled=runtime_settings.telemetry_enabled,
    )

    embedding_interface = EmbeddingInterface(
        adapter=EmbeddingAdapter(
            provider=runtime_settings.embedding_provider,
            dim=runtime_settings.embedding_dim,
        ),
        logger=logger,
        enabled=runtime_settings.embedding_enabled,
    )

    if runtime_settings.database_driver == "postgres":
        memory_repository = PostgresMemoryRepository(dsn=runtime_settings.postgres_dsn, logger=logger)
        profile_repository = PostgresProfileRepository(dsn=runtime_settings.postgres_dsn, logger=logger)
        goal_repository = PostgresGoalRepository(dsn=runtime_settings.postgres_dsn, logger=logger)
        skill_repository = PostgresSkillRepository(dsn=runtime_settings.postgres_dsn, logger=logger)
    else:
        memory_repository = SQLiteMemoryRepository(db_path=runtime_settings.memory_db_path, logger=logger)
        profile_repository = SQLiteProfileRepository(db_path=runtime_settings.memory_db_path, logger=logger)
        goal_repository = SQLiteGoalRepository(db_path=runtime_settings.memory_db_path, logger=logger)
        skill_repository = SQLiteSkillRepository(db_path=runtime_settings.memory_db_path, logger=logger)

    memory_store = MemoryStore(
        repository=memory_repository,
        logger=logger,
        embedding_interface=embedding_interface,
        importance_policy=MemoryImportancePolicy(),
        decay_policy=MemoryDecayPolicy(),
    )
    memory_interface = MemoryInterface(
        store=memory_store,
        retriever=MemoryRetriever(
            store=memory_store,
            logger=logger,
            embedding_interface=embedding_interface,
            ranking_policy=MemoryRankingPolicy(),
        ),
        summarizer=MemorySummarizer(store=memory_store, logger=logger),
        task_manager=task_manager,
        logger=logger,
        long_term_every_n_messages=runtime_settings.memory_long_term_every_n_messages,
        privacy_redaction_enabled=runtime_settings.privacy_redaction_enabled,
        privacy_remove_sensitive_context_keys=runtime_settings.privacy_remove_sensitive_context_keys,
    )

    profile_store = UserProfileStore(
        repository=profile_repository,
        logger=logger,
    )
    user_profile_interface = UserProfileInterface(store=profile_store, logger=logger)

    goal_store = GoalStore(
        repository=goal_repository,
        logger=logger,
    )
    goal_interface = GoalInterface(store=goal_store, logger=logger)
    relationship_memory_store = RelationshipMemoryStore(
        db_path=runtime_settings.memory_db_path,
        logger=logger,
    )
    relationship_state_engine = RelationshipStateEngine(
        db_path=runtime_settings.memory_db_path,
        logger=logger,
        memory_store=relationship_memory_store,
        relationship_memory_text_enabled=runtime_settings.relationship_memory_text_enabled,
        privacy_redaction_enabled=runtime_settings.privacy_redaction_enabled,
    )
    emotional_continuity_engine = EmotionalContinuityEngine(
        db_path=runtime_settings.memory_db_path,
        logger=logger,
    )
    safety_guardrails = SafetyGuardrails(logger=logger)
    personality_profile_service = PersonalityProfileService(
        db_path=runtime_settings.memory_db_path,
        logger=logger,
    )
    identity_core = IdentityCore(
        profile_service=personality_profile_service,
        logger=logger,
    )

    tool_interface = ToolInterface(
        registry=ToolRegistry(logger=logger),
        executor=ToolExecutor(
            logger=logger,
            max_timeout_seconds=runtime_settings.tool_max_timeout_seconds,
        ),
        logger=logger,
        enabled=runtime_settings.tool_execution_enabled,
    )
    skill_interface = SkillInterface(
        repository=skill_repository,
        tool_interface=tool_interface,
        logger=logger,
    )
    model_router = ModelRouter(logger=logger, enabled=runtime_settings.model_routing_enabled)
    resolved_model_adapter = model_adapter or ModelAdapter(
        logger=logger,
        backend=runtime_settings.model_backend,
        local_model_path=runtime_settings.local_model_path,
        local_model_threads=runtime_settings.local_model_threads,
        local_model_context_size=runtime_settings.local_model_context_size,
        local_model_max_tokens=runtime_settings.local_model_max_tokens,
    )
    response_style_engine = ResponseStyleEngine()
    response_match_predictor = ResponseMatchPredictor(
        model_adapter=resolved_model_adapter,
        logger=logger,
        model_judge_enabled=(
            runtime_settings.response_match_model_enabled
            and resolved_model_adapter.can_use_model_judge()
        ),
        model_judge_name=runtime_settings.response_match_model_name,
        model_judge_timeout_seconds=runtime_settings.response_match_model_timeout_seconds,
        model_judge_max_retries=runtime_settings.response_match_model_max_retries,
        model_match_threshold=runtime_settings.response_match_model_threshold,
    )
    directness_enforcer = DirectnessEnforcer()
    vagueness_detector = VaguenessDetector()
    topic_alignment_checker = TopicAlignmentChecker()
    human_coherence_scorer = HumanCoherenceScorer()
    execution_controller = ExecutionController(
        tool_interface=tool_interface,
        goal_interface=goal_interface,
        memory_interface=memory_interface,
        max_tool_calls=2,
    )
    orchestrator = BrainOrchestrator(
        intent_analyzer=IntentAnalyzer(),
        prompt_builder=PromptBuilder(),
        model_adapter=resolved_model_adapter,
        reflection_engine=ReflectionEngine(),
        response_formatter=ResponseFormatter(),
        memory_interface=memory_interface,
        user_profile_interface=user_profile_interface,
        execution_controller=execution_controller,
        model_router=model_router,
        planning_engine=PlanningEngine(),
        evaluation_engine=EvaluationEngine(),
        fallback_engine=FallbackEngine(),
        telemetry_engine=TelemetryEngine(exporter=telemetry_exporter),
        safety_classifier=SafetyClassifier(logger=logger),
        safe_web_lookup=SafeWebLookup(
            logger=logger,
            enabled=runtime_settings.internet_lookup_enabled,
            timeout_seconds=runtime_settings.internet_lookup_timeout_seconds,
            max_results=runtime_settings.internet_lookup_max_results,
            max_chars=runtime_settings.internet_lookup_max_chars,
        ),
        response_post_processor=ResponsePostProcessor(),
        response_style_engine=response_style_engine,
        response_match_predictor=response_match_predictor,
        directness_enforcer=directness_enforcer,
        vagueness_detector=vagueness_detector,
        topic_alignment_checker=topic_alignment_checker,
        human_coherence_scorer=human_coherence_scorer,
        identity_core=identity_core,
        relationship_state_engine=relationship_state_engine,
        emotional_continuity_engine=emotional_continuity_engine,
        safety_guardrails=safety_guardrails,
        skill_interface=skill_interface,
        logger=logger,
        settings_provider=lambda: runtime_settings,
    )
    brain = BrainInterface(
        orchestrator=orchestrator,
        model_adapter=resolved_model_adapter,
        memory_interface=memory_interface,
        user_profile_interface=user_profile_interface,
        model_router=model_router,
        tool_interface=tool_interface,
        goal_interface=goal_interface,
    )
    runtime_analyzer = RuntimeAnalyzer(
        logger=logger,
        enabled=runtime_settings.analyzer_enabled,
        file_path=runtime_settings.analyzer_file_path,
        poll_interval_seconds=runtime_settings.analyzer_poll_seconds,
        request_delta_threshold=runtime_settings.analyzer_request_delta_threshold,
        latency_delta_ms=runtime_settings.analyzer_latency_delta_ms,
        max_file_bytes=runtime_settings.analyzer_max_file_bytes,
        max_records=runtime_settings.analyzer_max_records,
        settings_provider=lambda: runtime_settings,
        brain_provider=lambda: brain,
        telemetry_exporter_provider=lambda: telemetry_exporter,
        skill_interface_provider=lambda: skill_interface,
    )
    return ServiceContainer(
        settings=runtime_settings,
        logger=logger,
        task_manager=task_manager,
        brain=brain,
        embedding_interface=embedding_interface,
        memory_interface=memory_interface,
        user_profile_interface=user_profile_interface,
        goal_interface=goal_interface,
        tool_interface=tool_interface,
        model_router=model_router,
        memory_store=memory_store,
        profile_store=profile_store,
        goal_store=goal_store,
        relationship_memory_store=relationship_memory_store,
        relationship_state_engine=relationship_state_engine,
        emotional_continuity_engine=emotional_continuity_engine,
        safety_guardrails=safety_guardrails,
        skill_interface=skill_interface,
        telemetry_exporter=telemetry_exporter,
        runtime_analyzer=runtime_analyzer,
        response_style_engine=response_style_engine,
        response_match_predictor=response_match_predictor,
        directness_enforcer=directness_enforcer,
        vagueness_detector=vagueness_detector,
        topic_alignment_checker=topic_alignment_checker,
        human_coherence_scorer=human_coherence_scorer,
    )


@lru_cache
def _cached_container() -> ServiceContainer:
    return create_container(settings=get_settings())


def get_brain() -> BrainInterface:
    return _cached_container().brain


def get_task_manager() -> TaskManager:
    return _cached_container().task_manager


def get_telemetry_exporter() -> TelemetryExporter:
    return _cached_container().telemetry_exporter


def get_skill_interface() -> SkillInterface:
    return _cached_container().skill_interface


def get_runtime_analyzer() -> RuntimeAnalyzer:
    return _cached_container().runtime_analyzer
