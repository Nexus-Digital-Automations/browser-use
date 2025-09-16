"""
Parlant Integration Service for Browser-Use

Provides conversational AI validation for ALL Browser-Use browser automation,
web interaction, and AI agent functions. Implements function-level Parlant 
integration to ensure AI execution precision and safety guardrails.

This module wraps every browser automation, web interaction, and task execution
function with Parlant's conversational AI validation system for maximum control
and safety in web automation scenarios.

@author Parlant Integration Team
@since 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp


class ParlantIntegrationService:
	"""
	Parlant Integration Service for Browser-Use
	
	Provides conversational AI validation for all browser automation, web 
	interaction, and AI agent functions. Ensures safety guardrails and intent
	verification for every browser operation and web automation task.
	
	Example:
		service = ParlantIntegrationService()
		result = await service.validate_operation(
			operation="browser_action",
			context={"action": "click", "element": "button", "url": "example.com"},
			user_intent="Click submit button to complete form submission"
		)
	"""
	
	# Parlant Configuration Constants
	PARLANT_API_BASE_URL = os.getenv('PARLANT_API_BASE_URL', 'http://localhost:8000')
	PARLANT_API_TIMEOUT = float(os.getenv('PARLANT_API_TIMEOUT_MS', '10000')) / 1000.0
	PARLANT_ENABLED = os.getenv('PARLANT_ENABLED', 'true').lower() == 'true'
	PARLANT_CACHE_ENABLED = os.getenv('PARLANT_CACHE_ENABLED', 'true').lower() == 'true'
	PARLANT_CACHE_MAX_AGE = float(os.getenv('PARLANT_CACHE_MAX_AGE_MS', '300000')) / 1000.0
	
	# Risk Level Definitions for Browser-Use Operations
	RISK_LEVELS = {
		'low': [
			'get_page_info', 'screenshot', 'get_url', 'get_title', 'get_dom_snapshot',
			'scroll_info', 'element_visibility', 'page_status', 'health_check'
		],
		'medium': [
			'scroll', 'navigate', 'wait_for_element', 'search_elements', 'highlight_element',
			'extract_text', 'get_page_content', 'refresh_page', 'go_back', 'go_forward'
		],
		'high': [
			'click', 'type', 'select', 'drag_and_drop', 'upload_file', 'download_file',
			'submit_form', 'execute_script', 'set_cookies', 'clear_cache', 'browser_action'
		],
		'critical': [
			'run_task', 'execute_automation', 'mass_actions', 'system_interaction',
			'credential_input', 'financial_transaction', 'admin_action', 'bulk_operation'
		]
	}
	
	# Operation Categories for Browser-Use
	OPERATION_CATEGORIES = {
		'browser_automation': [
			'click', 'type', 'select', 'scroll', 'navigate', 'upload_file',
			'download_file', 'drag_and_drop', 'submit_form', 'browser_action'
		],
		'web_interaction': [
			'extract_text', 'search_elements', 'wait_for_element', 'highlight_element',
			'get_page_content', 'execute_script', 'interact_with_element'
		],
		'task_execution': [
			'run_task', 'run_prompt_mode', 'execute_automation', 'agent_run',
			'complete_objective', 'process_workflow'
		],
		'session_management': [
			'create_session', 'close_session', 'switch_tab', 'manage_cookies',
			'clear_cache', 'set_viewport', 'configure_browser'
		],
		'ai_integration': [
			'llm_decision', 'analyze_page', 'generate_actions', 'process_response',
			'interpret_content', 'plan_automation'
		]
	}
	
	def __init__(self):
		"""Initialize Parlant Integration Service for Browser-Use"""
		self.logger = self._setup_logging()
		self.cache = {}
		self.metrics = self._initialize_metrics()
		self.operation_counter = 0
		self.conversation_context = {}
		self._thread_pool = ThreadPoolExecutor(max_workers=5)
		
		self._log_service_initialization()
	
	def _setup_logging(self) -> logging.Logger:
		"""Setup structured logging for Parlant operations"""
		logger = logging.getLogger('ParlantIntegrationBrowserUse')
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter(
				'[%(asctime)s] %(levelname)s %(name)s: %(message)s'
			)
			handler.setFormatter(formatter)
			logger.addHandler(handler)
			logger.setLevel(getattr(logging, os.getenv('PARLANT_LOG_LEVEL', 'INFO').upper()))
		return logger
	
	def _initialize_metrics(self) -> Dict[str, Any]:
		"""Initialize performance metrics tracking"""
		return {
			'total_validations': 0,
			'successful_validations': 0,
			'failed_validations': 0,
			'average_response_time': 0.0,
			'cache_hits': 0,
			'cache_misses': 0,
			'high_risk_operations': 0,
			'blocked_operations': 0,
			'browser_actions': 0,
			'task_executions': 0
		}
	
	async def validate_operation(
		self,
		operation: str,
		context: Dict[str, Any] = None,
		user_intent: str = None,
		risk_assessment: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""
		Core validation method for Browser-Use operations
		
		Validates browser automation, web interactions, and AI agent tasks
		through Parlant's conversational AI engine with safety guardrails.
		
		Args:
			operation: The operation being performed
			context: Operation context (actions, elements, URLs, tasks)
			user_intent: Natural language description of user intent
			risk_assessment: Optional custom risk assessment
			
		Returns:
			Dict containing validation result with approval status and metadata
			
		Example:
			result = await validate_operation(
				operation="browser_action",
				context={
					"action": "click",
					"element": "submit_button",
					"url": "https://form.example.com",
					"task_description": "Submit contact form"
				},
				user_intent="Click submit button to send contact form data"
			)
		"""
		operation_id = self._generate_operation_id()
		start_time = time.time()
		
		self._log_validation_start(operation_id, operation, context, user_intent)
		
		if not self.PARLANT_ENABLED:
			return self._bypass_result(operation_id, "Parlant disabled")
		
		try:
			# Check cache for performance optimization
			cache_key = self._generate_cache_key(operation, context, user_intent)
			cached_result = self._get_cached_result(cache_key)
			if cached_result:
				return self._add_operation_metadata(cached_result, operation_id, start_time)
			
			# Perform risk assessment
			risk_level = risk_assessment or self._assess_operation_risk(operation, context)
			
			# Build validation request
			validation_request = self._build_validation_request(
				operation, context, user_intent, risk_level, operation_id
			)
			
			# Execute conversational validation
			validation_result = await self._execute_parlant_validation(validation_request, operation_id)
			
			# Process and enhance result
			enhanced_result = self._process_validation_result(validation_result, risk_level, operation_id)
			
			# Cache successful validations
			if enhanced_result['approved']:
				self._cache_validation_result(cache_key, enhanced_result)
			
			# Update metrics
			self._record_validation_metrics(operation_id, enhanced_result, time.time() - start_time)
			
			self._log_validation_completion(operation_id, enhanced_result)
			return self._add_operation_metadata(enhanced_result, operation_id, start_time)
			
		except Exception as e:
			return self._handle_validation_error(e, operation_id, operation, context)
	
	async def validate_browser_action(
		self,
		action: str,
		element_info: Dict[str, Any] = None,
		url: str = None,
		task_context: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""
		Specialized validation for browser automation actions
		
		Validates browser actions like click, type, scroll, navigate before execution.
		
		Args:
			action: Browser action being performed (click, type, scroll, etc.)
			element_info: Information about target element
			url: Current or target URL
			task_context: Task execution context
			
		Returns:
			Validation result for browser action
		"""
		context = {
			'action': action,
			'element_info': element_info or {},
			'url': url,
			'domain': urlparse(url).netloc if url else None,
			'action_category': self._categorize_action(action),
			'task_context': task_context or {},
			'requires_user_input': action.lower() in ['type', 'upload', 'select'],
			'modifies_page_state': action.lower() in ['click', 'submit', 'delete', 'clear']
		}
		
		return await self.validate_operation(
			operation='browser_action',
			context=context,
			user_intent=f"Execute browser {action} action on {url or 'current page'}"
		)
	
	async def validate_task_execution(
		self,
		task_description: str,
		execution_context: Dict[str, Any] = None,
		agent_config: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""
		Specialized validation for AI agent task execution
		
		Validates high-level task execution and automation workflows.
		
		Args:
			task_description: Description of task to execute
			execution_context: Task execution environment and settings
			agent_config: AI agent configuration and capabilities
			
		Returns:
			Validation result for task execution
		"""
		context = {
			'task_description': task_description,
			'task_length': len(task_description),
			'execution_context': execution_context or {},
			'agent_config': agent_config or {},
			'estimated_complexity': self._estimate_task_complexity(task_description),
			'requires_web_interaction': any(keyword in task_description.lower() for keyword in [
				'click', 'type', 'navigate', 'submit', 'form', 'button', 'link'
			]),
			'involves_sensitive_data': any(keyword in task_description.lower() for keyword in [
				'login', 'password', 'credit', 'payment', 'personal', 'private'
			])
		}
		
		return await self.validate_operation(
			operation='task_execution',
			context=context,
			user_intent=f"Execute automation task: {task_description[:100]}{'...' if len(task_description) > 100 else ''}"
		)
	
	async def validate_web_interaction(
		self,
		interaction_type: str,
		target_info: Dict[str, Any] = None,
		page_context: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""
		Specialized validation for web page interactions
		
		Validates web interactions like element extraction, content analysis, etc.
		
		Args:
			interaction_type: Type of web interaction (extract, analyze, search)
			target_info: Information about interaction target
			page_context: Current page context and state
			
		Returns:
			Validation result for web interaction
		"""
		context = {
			'interaction_type': interaction_type,
			'target_info': target_info or {},
			'page_context': page_context or {},
			'current_url': page_context.get('url') if page_context else None,
			'page_title': page_context.get('title') if page_context else None,
			'is_sensitive_site': self._is_sensitive_domain(page_context.get('url')) if page_context else False,
			'interaction_scope': 'read_only' if interaction_type in ['extract', 'analyze', 'search'] else 'interactive'
		}
		
		return await self.validate_operation(
			operation='web_interaction',
			context=context,
			user_intent=f"Perform {interaction_type} web interaction"
		)
	
	async def validate_session_management(
		self,
		session_action: str,
		session_info: Dict[str, Any] = None,
		browser_config: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""
		Specialized validation for browser session management
		
		Validates session operations like creating, closing, configuring browser sessions.
		
		Args:
			session_action: Session action (create, close, configure, etc.)
			session_info: Current session information
			browser_config: Browser configuration parameters
			
		Returns:
			Validation result for session management
		"""
		context = {
			'session_action': session_action,
			'session_info': session_info or {},
			'browser_config': browser_config or {},
			'session_count': session_info.get('active_sessions', 0) if session_info else 0,
			'requires_permissions': session_action in ['create', 'configure', 'reset'],
			'affects_security': session_action in ['configure', 'clear_cookies', 'clear_cache']
		}
		
		return await self.validate_operation(
			operation='session_management',
			context=context,
			user_intent=f"Perform browser session {session_action} operation"
		)
	
	def get_health_status(self) -> Dict[str, Any]:
		"""
		Get comprehensive health status of Parlant integration
		
		Returns:
			Health status including API connectivity, performance metrics
		"""
		return {
			'parlant_enabled': self.PARLANT_ENABLED,
			'api_connectivity': asyncio.run(self._check_api_connectivity()),
			'cache_status': self._check_cache_status(),
			'performance_metrics': self._get_performance_metrics(),
			'recent_validations': self._get_recent_validation_stats(),
			'service_uptime': time.time(),
			'timestamp': datetime.now().isoformat(),
			'browser_use_integration': 'active'
		}
	
	def _assess_operation_risk(self, operation: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Assess risk level of Browser-Use operation"""
		base_risk = self._determine_base_risk_level(operation)
		risk_factors = self._analyze_risk_factors(operation, context)
		
		# Adjust risk based on context
		if context:
			# URL-based risk assessment
			if context.get('url'):
				url = context['url']
				if self._is_sensitive_domain(url):
					base_risk = 'high' if base_risk in ['low', 'medium'] else base_risk
				if any(sensitive in url.lower() for sensitive in [
					'bank', 'payment', 'login', 'admin', 'secure'
				]):
					base_risk = 'critical'
			
			# Action-based risk escalation
			if context.get('action'):
				action = context['action'].lower()
				if action in ['type', 'input'] and any(field in str(context).lower() for field in [
					'password', 'credit', 'ssn', 'personal'
				]):
					base_risk = 'critical'
				elif action in ['submit', 'delete', 'clear'] and base_risk in ['low', 'medium']:
					base_risk = 'high'
			
			# Task complexity risk
			if context.get('task_description'):
				if any(complex_keyword in context['task_description'].lower() for complex_keyword in [
					'automate', 'bulk', 'mass', 'multiple', 'sequence'
				]):
					base_risk = 'high' if base_risk in ['low', 'medium'] else base_risk
		
		return {
			'level': base_risk,
			'factors': risk_factors,
			'requires_approval': base_risk in ['high', 'critical'],
			'assessment_time': datetime.now().isoformat()
		}
	
	def _determine_base_risk_level(self, operation: str) -> str:
		"""Map operations to base risk levels"""
		for level, operations in self.RISK_LEVELS.items():
			if operation in operations:
				return level
		
		# Check operation patterns
		if any(keyword in operation.lower() for keyword in ['execute', 'run', 'automation']):
			return 'high'
		elif any(keyword in operation.lower() for keyword in ['admin', 'system', 'critical']):
			return 'critical'
		elif any(keyword in operation.lower() for keyword in ['click', 'type', 'submit']):
			return 'medium'
		elif any(keyword in operation.lower() for keyword in ['get', 'read', 'view']):
			return 'low'
		
		return 'medium'  # Default for unmapped operations
	
	def _analyze_risk_factors(self, operation: str, context: Dict[str, Any]) -> List[str]:
		"""Identify specific risk factors based on operation context"""
		factors = []
		
		if context:
			# URL and domain factors
			if context.get('url'):
				url = context['url']
				if self._is_sensitive_domain(url):
					factors.append('sensitive_domain')
				if url.startswith('https://'):
					factors.append('secure_connection')
				else:
					factors.append('insecure_connection')
			
			# Action-specific factors
			if context.get('action'):
				action = context['action'].lower()
				if action in ['type', 'input']:
					factors.append('user_input_required')
				if action in ['submit', 'click']:
					factors.append('state_modification')
				if action in ['upload', 'download']:
					factors.append('file_operation')
			
			# Task complexity factors
			if context.get('task_description'):
				task = context['task_description'].lower()
				if len(task) > 200:
					factors.append('complex_task')
				if any(auth_keyword in task for auth_keyword in ['login', 'signin', 'authenticate']):
					factors.append('authentication_required')
				if any(finance_keyword in task for finance_keyword in ['payment', 'purchase', 'transaction']):
					factors.append('financial_operation')
			
			# Browser session factors
			if context.get('session_info'):
				if context['session_info'].get('active_sessions', 0) > 5:
					factors.append('multiple_sessions')
			
			# Element interaction factors
			if context.get('element_info'):
				element = context['element_info']
				if element.get('type') in ['password', 'email', 'tel']:
					factors.append('sensitive_form_field')
				if element.get('required'):
					factors.append('required_field')
		
		# Operation-specific factors
		if 'automation' in operation.lower():
			factors.append('automated_operation')
		if 'bulk' in operation.lower() or 'mass' in operation.lower():
			factors.append('bulk_operation')
		if 'admin' in operation.lower():
			factors.append('administrative_operation')
		
		return factors
	
	def _categorize_action(self, action: str) -> str:
		"""Categorize browser action by type"""
		action_lower = action.lower()
		
		if action_lower in ['click', 'tap', 'double_click']:
			return 'navigation_action'
		elif action_lower in ['type', 'input', 'fill']:
			return 'input_action'
		elif action_lower in ['scroll', 'swipe', 'drag']:
			return 'viewport_action'
		elif action_lower in ['select', 'choose', 'pick']:
			return 'selection_action'
		elif action_lower in ['submit', 'send', 'post']:
			return 'submission_action'
		elif action_lower in ['navigate', 'goto', 'visit']:
			return 'navigation_action'
		else:
			return 'general_action'
	
	def _estimate_task_complexity(self, task_description: str) -> str:
		"""Estimate task complexity based on description"""
		description_lower = task_description.lower()
		word_count = len(task_description.split())
		
		# Count complexity indicators
		complexity_indicators = sum([
			description_lower.count('and'),
			description_lower.count('then'),
			description_lower.count('after'),
			description_lower.count('if'),
			description_lower.count('when'),
			description_lower.count('multiple'),
			description_lower.count('several')
		])
		
		if word_count > 50 or complexity_indicators > 3:
			return 'high'
		elif word_count > 20 or complexity_indicators > 1:
			return 'medium'
		else:
			return 'low'
	
	def _is_sensitive_domain(self, url: str) -> bool:
		"""Check if URL is from a sensitive domain"""
		if not url:
			return False
		
		try:
			domain = urlparse(url).netloc.lower()
			sensitive_domains = [
				'bank', 'finance', 'payment', 'paypal', 'stripe',
				'login', 'auth', 'admin', 'secure', 'private',
				'government', 'medical', 'health', 'insurance'
			]
			return any(sensitive in domain for sensitive in sensitive_domains)
		except Exception:
			return False
	
	def _build_validation_request(
		self,
		operation: str,
		context: Dict[str, Any],
		user_intent: str,
		risk_assessment: Dict[str, Any],
		operation_id: str
	) -> Dict[str, Any]:
		"""Build validation request payload for Parlant API"""
		return {
			'operation_id': operation_id,
			'operation': operation,
			'context': self._sanitize_context(context),
			'user_intent': user_intent or f"Perform {operation} operation",
			'risk_assessment': risk_assessment,
			'system_info': {
				'service': 'browser-use',
				'version': '1.0.0',
				'environment': os.getenv('ENVIRONMENT', 'development'),
				'timestamp': datetime.now().isoformat(),
				'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
			},
			'validation_settings': {
				'require_approval': risk_assessment['requires_approval'],
				'timeout_ms': int(self.PARLANT_API_TIMEOUT * 1000),
				'cache_enabled': self.PARLANT_CACHE_ENABLED
			}
		}
	
	async def _execute_parlant_validation(
		self,
		request_payload: Dict[str, Any],
		operation_id: str
	) -> Dict[str, Any]:
		"""Execute validation request to Parlant API"""
		self.logger.debug(f"[{operation_id}] Executing Parlant validation", extra={
			'operation': request_payload['operation'],
			'risk_level': request_payload['risk_assessment']['level']
		})
		
		try:
			timeout = aiohttp.ClientTimeout(total=self.PARLANT_API_TIMEOUT)
			
			async with aiohttp.ClientSession(timeout=timeout) as session:
				headers = {
					'Content-Type': 'application/json',
					'Accept': 'application/json',
					'X-Operation-ID': operation_id,
					'User-Agent': 'BrowserUse-Parlant-Integration/1.0.0'
				}
				
				if os.getenv('PARLANT_API_KEY'):
					headers['Authorization'] = f"Bearer {os.getenv('PARLANT_API_KEY')}"
				
				async with session.post(
					f"{self.PARLANT_API_BASE_URL}/api/v1/validate",
					json=request_payload,
					headers=headers
				) as response:
					response.raise_for_status()
					return await response.json()
					
		except Exception as e:
			self.logger.error(f"[{operation_id}] Parlant API request failed: {e}")
			raise
	
	def _process_validation_result(
		self,
		validation_result: Dict[str, Any],
		risk_assessment: Dict[str, Any],
		operation_id: str
	) -> Dict[str, Any]:
		"""Process and enhance validation results from Parlant API"""
		return {
			'approved': validation_result.get('approved', False),
			'confidence': validation_result.get('confidence', 0.0),
			'reasoning': validation_result.get('reasoning', 'No reasoning provided'),
			'risk_level': risk_assessment['level'],
			'operation_id': operation_id,
			'validation_metadata': {
				'parlant_session_id': validation_result.get('session_id'),
				'response_time_ms': validation_result.get('response_time_ms'),
				'model_version': validation_result.get('model_version'),
				'validation_timestamp': datetime.now().isoformat()
			},
			'recommendations': validation_result.get('recommendations', []),
			'warnings': validation_result.get('warnings', [])
		}
	
	def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Remove sensitive data from context for API transmission"""
		if not context:
			return {}
		
		sanitized = context.copy()
		
		# Remove sensitive fields
		sensitive_keys = ['password', 'secret', 'token', 'api_key', 'credential', 'private_key', 'auth']
		for key in sensitive_keys:
			if key in sanitized:
				sanitized[key] = '[REDACTED]'
		
		# Sanitize nested dictionaries
		for key, value in sanitized.items():
			if isinstance(value, dict):
				sanitized[key] = self._sanitize_context(value)
			elif isinstance(value, str) and len(value) > 1000:
				sanitized[key] = value[:997] + '...'
		
		return sanitized
	
	def _generate_cache_key(self, operation: str, context: Dict[str, Any], user_intent: str) -> str:
		"""Generate deterministic cache key"""
		key_data = {
			'operation': operation,
			'context_hash': hashlib.sha256(json.dumps(context or {}, sort_keys=True).encode()).hexdigest(),
			'intent_hash': hashlib.sha256((user_intent or '').encode()).hexdigest()
		}
		return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
	
	def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
		"""Retrieve cached validation result if available and valid"""
		if not self.PARLANT_CACHE_ENABLED or cache_key not in self.cache:
			self.metrics['cache_misses'] += 1
			return None
		
		cached_item = self.cache[cache_key]
		if time.time() - cached_item['timestamp'] > self.PARLANT_CACHE_MAX_AGE:
			del self.cache[cache_key]
			self.metrics['cache_misses'] += 1
			return None
		
		self.metrics['cache_hits'] += 1
		self.logger.debug(f"Cache hit for key: {cache_key[:16]}...")
		return cached_item['result']
	
	def _cache_validation_result(self, cache_key: str, result: Dict[str, Any]) -> None:
		"""Store validation result in cache"""
		if not self.PARLANT_CACHE_ENABLED:
			return
		
		self.cache[cache_key] = {
			'result': result,
			'timestamp': time.time()
		}
		
		# Simple cache cleanup
		if len(self.cache) > 1000:
			oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
			del self.cache[oldest_key]
	
	def _generate_operation_id(self) -> str:
		"""Generate unique operation identifier"""
		self.operation_counter += 1
		return f"browser_use_parlant_{int(time.time())}_{self.operation_counter}_{threading.current_thread().ident}"
	
	def _bypass_result(self, operation_id: str, reason: str) -> Dict[str, Any]:
		"""Return approval result when Parlant is disabled"""
		return {
			'approved': True,
			'bypassed': True,
			'bypass_reason': reason,
			'operation_id': operation_id,
			'confidence': 1.0,
			'reasoning': f"Parlant validation bypassed: {reason}",
			'validation_metadata': {
				'bypass_timestamp': datetime.now().isoformat()
			}
		}
	
	def _add_operation_metadata(
		self,
		result: Dict[str, Any],
		operation_id: str,
		start_time: float
	) -> Dict[str, Any]:
		"""Add timing and operation metadata to validation results"""
		result.update({
			'operation_id': operation_id,
			'total_duration_ms': round((time.time() - start_time) * 1000, 2),
			'processed_at': datetime.now().isoformat()
		})
		return result
	
	def _handle_validation_error(
		self,
		error: Exception,
		operation_id: str,
		operation: str,
		context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Handle errors during validation process"""
		self.logger.error(f"[{operation_id}] Validation failed: {error}", extra={
			'operation': operation,
			'context_keys': list(context.keys()) if context else [],
			'error_type': type(error).__name__
		})
		
		self.metrics['failed_validations'] += 1
		
		# Safe default based on risk level
		risk_level = self._determine_base_risk_level(operation)
		safe_default = risk_level not in ['high', 'critical']
		
		return {
			'approved': safe_default,
			'error': True,
			'error_message': str(error),
			'operation_id': operation_id,
			'confidence': 0.0,
			'reasoning': f"Validation failed due to error: {error}",
			'validation_metadata': {
				'error_timestamp': datetime.now().isoformat(),
				'error_class': type(error).__name__
			}
		}
	
	def _record_validation_metrics(
		self,
		operation_id: str,
		result: Dict[str, Any],
		duration: float
	) -> None:
		"""Update performance metrics after validation completion"""
		self.metrics['total_validations'] += 1
		
		if result['approved']:
			self.metrics['successful_validations'] += 1
		else:
			self.metrics['failed_validations'] += 1
			if result.get('risk_level') in ['high', 'critical']:
				self.metrics['blocked_operations'] += 1
		
		if result.get('risk_level') in ['high', 'critical']:
			self.metrics['high_risk_operations'] += 1
		
		# Track specific operation types
		if 'browser_action' in operation_id:
			self.metrics['browser_actions'] += 1
		elif 'task_execution' in operation_id:
			self.metrics['task_executions'] += 1
		
		# Update average response time
		current_avg = self.metrics['average_response_time']
		total_count = self.metrics['total_validations']
		self.metrics['average_response_time'] = ((current_avg * (total_count - 1)) + duration) / total_count
	
	async def _check_api_connectivity(self) -> Dict[str, Any]:
		"""Test connection to Parlant API"""
		try:
			timeout = aiohttp.ClientTimeout(total=5)
			async with aiohttp.ClientSession(timeout=timeout) as session:
				start_time = time.time()
				async with session.get(f"{self.PARLANT_API_BASE_URL}/api/v1/health") as response:
					response_time = (time.time() - start_time) * 1000
					return {
						'connected': response.status == 200,
						'response_time_ms': response_time,
						'last_check': datetime.now().isoformat()
					}
		except Exception as e:
			return {
				'connected': False,
				'error': str(e),
				'last_check': datetime.now().isoformat()
			}
	
	def _check_cache_status(self) -> Dict[str, Any]:
		"""Check cache functionality and performance"""
		if not self.PARLANT_CACHE_ENABLED:
			return {'enabled': False}
		
		hit_rate = 0.0
		total_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
		if total_requests > 0:
			hit_rate = (self.metrics['cache_hits'] / total_requests) * 100
		
		return {
			'enabled': True,
			'entries': len(self.cache),
			'hits': self.metrics['cache_hits'],
			'misses': self.metrics['cache_misses'],
			'hit_rate_percent': round(hit_rate, 2)
		}
	
	def _get_performance_metrics(self) -> Dict[str, Any]:
		"""Get current performance metrics"""
		success_rate = 0.0
		if self.metrics['total_validations'] > 0:
			success_rate = (self.metrics['successful_validations'] / self.metrics['total_validations']) * 100
		
		return {
			**self.metrics,
			'success_rate_percent': round(success_rate, 2),
			'cache_hit_rate_percent': round(
				(self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])) * 100, 2
			)
		}
	
	def _get_recent_validation_stats(self) -> Dict[str, Any]:
		"""Get recent validation statistics"""
		return {
			'recent_validations': self.metrics['total_validations'],
			'recent_success_rate': round(
				(self.metrics['successful_validations'] / max(1, self.metrics['total_validations'])) * 100, 2
			),
			'average_response_time_ms': round(self.metrics['average_response_time'] * 1000, 2),
			'high_risk_operations': self.metrics['high_risk_operations'],
			'blocked_operations': self.metrics['blocked_operations'],
			'browser_actions': self.metrics['browser_actions'],
			'task_executions': self.metrics['task_executions']
		}
	
	def _log_service_initialization(self) -> None:
		"""Log service startup information"""
		self.logger.info("Parlant Integration Service initialized for Browser-Use", extra={
			'parlant_enabled': self.PARLANT_ENABLED,
			'api_base_url': self.PARLANT_API_BASE_URL,
			'cache_enabled': self.PARLANT_CACHE_ENABLED,
			'timeout_seconds': self.PARLANT_API_TIMEOUT,
			'service': 'browser-use'
		})
	
	def _log_validation_start(
		self,
		operation_id: str,
		operation: str,
		context: Dict[str, Any],
		user_intent: str
	) -> None:
		"""Log validation operation start"""
		self.logger.info(f"[{operation_id}] Validation started", extra={
			'operation': operation,
			'context_keys': list(context.keys()) if context else [],
			'user_intent': user_intent,
			'timestamp': datetime.now().isoformat()
		})
	
	def _log_validation_completion(
		self,
		operation_id: str,
		result: Dict[str, Any]
	) -> None:
		"""Log validation operation completion"""
		self.logger.info(f"[{operation_id}] Validation completed", extra={
			'approved': result['approved'],
			'confidence': result['confidence'],
			'risk_level': result['risk_level'],
			'bypassed': result.get('bypassed', False),
			'timestamp': datetime.now().isoformat()
		})


# Global service instance
_parlant_service: Optional[ParlantIntegrationService] = None


def get_parlant_service() -> ParlantIntegrationService:
	"""Get global Parlant service instance (singleton pattern)"""
	global _parlant_service
	if _parlant_service is None:
		_parlant_service = ParlantIntegrationService()
	return _parlant_service


def parlant_validate(operation_type: str = None):
	"""
	Decorator for function-level Parlant validation
	
	Wraps Browser-Use functions with conversational AI validation.
	
	Args:
		operation_type: Type of operation for risk assessment
		
	Example:
		@parlant_validate("browser_action")
		async def click_element(element_id, url):
			# Function implementation
			pass
	"""
	def decorator(func):
		@wraps(func)
		async def async_wrapper(*args, **kwargs):
			service = get_parlant_service()
			
			# Extract context from function arguments
			context = {
				'function_name': func.__name__,
				'args_count': len(args),
				'kwargs_keys': list(kwargs.keys()) if kwargs else [],
				'module': func.__module__
			}
			
			# Add specific context based on function signature and Browser-Use patterns
			if 'action' in kwargs:
				context['action'] = kwargs['action']
			if 'url' in kwargs:
				context['url'] = kwargs['url']
			if 'element' in kwargs or 'element_id' in kwargs:
				context['element_interaction'] = True
			if 'task' in kwargs or 'task_description' in kwargs:
				context['task_description'] = kwargs.get('task') or kwargs.get('task_description')
			if len(args) > 0 and isinstance(args[0], str):
				context['primary_input'] = args[0][:200]  # First 200 chars
			
			# Perform validation
			validation_result = await service.validate_operation(
				operation=operation_type or func.__name__,
				context=context,
				user_intent=f"Execute {func.__name__} function"
			)
			
			if not validation_result['approved']:
				raise PermissionError(
					f"Parlant validation blocked {func.__name__}: {validation_result['reasoning']}"
				)
			
			# Execute original function
			if asyncio.iscoroutinefunction(func):
				return await func(*args, **kwargs)
			else:
				return func(*args, **kwargs)
		
		@wraps(func)
		def sync_wrapper(*args, **kwargs):
			# For synchronous functions, run validation in event loop
			try:
				loop = asyncio.get_event_loop()
			except RuntimeError:
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
			
			return loop.run_until_complete(async_wrapper(*args, **kwargs))
		
		return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
	
	return decorator