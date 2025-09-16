"""
Parlant Validated Functions for Browser-Use

Provides function-level Parlant validation for ALL Browser-Use operations including
agent task execution, browser automation, web interactions, and session management.

This module implements comprehensive conversational AI validation wrappers for
every Browser-Use function to ensure maximum safety and intent verification.

@author Parlant Integration Team
@since 1.0.0
"""

import asyncio
import functools
import logging
from datetime import datetime
from typing import Any, Callable, Dict

from browser_use.parlant_integration import get_parlant_service


class ParlantValidatedBrowserUse:
	"""
	Mixin class that provides Parlant validation for Browser-Use classes
	
	Wraps all browser automation, web interaction, and AI agent functions
	with conversational AI validation to ensure safety and intent verification.
	
	Example:
		class MyAgent(Agent, ParlantValidatedBrowserUse):
			pass
			
		agent = MyAgent(task="Navigate to example.com")
		# All agent methods now include Parlant validation
	"""
	
	def __init__(self, *args, **kwargs):
		"""Initialize Parlant validated Browser-Use with service integration"""
		super().__init__(*args, **kwargs)
		self.parlant_service = get_parlant_service()
		self.logger = logging.getLogger(f'ParlantValidated{self.__class__.__name__}')
		
		self.logger.info(f"Parlant validation enabled for {self.__class__.__name__}", extra={
			'service': 'browser-use',
			'class': self.__class__.__name__,
			'parlant_enabled': self.parlant_service.PARLANT_ENABLED,
			'timestamp': datetime.now().isoformat()
		})


def parlant_validated_agent_run(original_method: Callable) -> Callable:
	"""
	Parlant validation wrapper for Agent.run() method
	
	Validates high-level agent task execution with comprehensive risk assessment.
	"""
	@functools.wraps(original_method)
	async def wrapper(self, max_steps: int = 100, **kwargs) -> Any:
		parlant_service = get_parlant_service()
		
		# Build comprehensive task execution context
		context = {
			'method': 'agent_run',
			'task_description': getattr(self, 'task', 'Unknown task'),
			'max_steps': max_steps,
			'agent_id': getattr(self, 'session_id', 'unknown'),
			'browser_session_active': hasattr(self, 'browser_session') and self.browser_session is not None,
			'has_tools': hasattr(self, 'tools') and self.tools is not None,
			'agent_settings': getattr(self, 'settings', {}).__dict__ if hasattr(self, 'settings') else {},
			'execution_type': 'autonomous_agent_run'
		}
		
		# Validate task execution
		validation_result = await parlant_service.validate_task_execution(
			task_description=context['task_description'],
			execution_context=context,
			agent_config={
				'max_steps': max_steps,
				'class_name': self.__class__.__name__
			}
		)
		
		if not validation_result['approved']:
			raise PermissionError(
				f"Parlant validation blocked agent task execution: {validation_result['reasoning']}"
			)
		
		return await original_method(self, max_steps=max_steps, **kwargs)
	
	return wrapper


def parlant_validated_agent_step(original_method: Callable) -> Callable:
	"""
	Parlant validation wrapper for Agent.step() method
	
	Validates individual agent execution steps with context awareness.
	"""
	@functools.wraps(original_method)
	async def wrapper(self, step_info=None, **kwargs) -> Any:
		parlant_service = get_parlant_service()
		
		# Build step execution context
		context = {
			'method': 'agent_step',
			'step_info': step_info.__dict__ if step_info else {},
			'current_step': getattr(self, 'state', {}).step_number if hasattr(self, 'state') else 0,
			'agent_state': getattr(self, 'state', {}).__dict__ if hasattr(self, 'state') else {},
			'has_browser_state': hasattr(self, 'browser_session'),
			'execution_type': 'agent_step'
		}
		
		# Validate step execution
		validation_result = await parlant_service.validate_operation(
			operation='agent_step',
			context=context,
			user_intent=f"Execute agent step {context['current_step']}"
		)
		
		if not validation_result['approved']:
			raise PermissionError(
				f"Parlant validation blocked agent step: {validation_result['reasoning']}"
			)
		
		return await original_method(self, step_info=step_info, **kwargs)
	
	return wrapper


def parlant_validated_browser_action(action_type: str) -> Callable:
	"""
	Parlant validation decorator factory for browser actions
	
	Creates validation wrappers for specific browser action types.
	
	Args:
		action_type: Type of browser action (click, type, navigate, etc.)
	"""
	def decorator(original_method: Callable) -> Callable:
		@functools.wraps(original_method)
		async def wrapper(*args, **kwargs) -> Any:
			parlant_service = get_parlant_service()
			
			# Extract browser session and action parameters
			browser_session = None
			action_params = {}
			
			# Find browser session in arguments
			for arg in args:
				if hasattr(arg, 'url') or hasattr(arg, 'cdp_url'):  # Browser session detection
					browser_session = arg
					break
			
			if not browser_session:
				for key, value in kwargs.items():
					if hasattr(value, 'url') or hasattr(value, 'cdp_url'):
						browser_session = value
						break
			
			# Extract action parameters from first argument if it has the action data
			if len(args) > 0 and hasattr(args[0], '__dict__'):
				action_params = args[0].__dict__
			
			# Build browser action context
			context = {
				'action': action_type,
				'action_params': action_params,
				'browser_session_id': getattr(browser_session, 'id', 'unknown') if browser_session else None,
				'current_url': getattr(browser_session, 'current_url', None) if browser_session else None,
				'method_name': original_method.__name__,
				'args_count': len(args),
				'kwargs_keys': list(kwargs.keys())
			}
			
			# Add specific context based on action type
			if action_type == 'click' and 'index' in action_params:
				context['element_index'] = action_params['index']
				context['element_interaction'] = True
			elif action_type == 'type' and 'text' in action_params:
				context['text_input'] = True
				context['input_length'] = len(str(action_params['text']))
			elif action_type == 'navigate' and 'url' in action_params:
				context['url'] = action_params['url']
				context['navigation'] = True
			elif action_type == 'upload' and 'file_paths' in action_params:
				context['file_operation'] = True
				context['file_count'] = len(action_params.get('file_paths', []))
			
			# Validate browser action
			validation_result = await parlant_service.validate_browser_action(
				action=action_type,
				element_info=action_params,
				url=context.get('current_url'),
				task_context=context
			)
			
			if not validation_result['approved']:
				raise PermissionError(
					f"Parlant validation blocked {action_type} action: {validation_result['reasoning']}"
				)
			
			return await original_method(*args, **kwargs)
		
		return wrapper
	return decorator


def parlant_validated_web_interaction(interaction_type: str) -> Callable:
	"""
	Parlant validation decorator factory for web interactions
	
	Creates validation wrappers for web interaction operations.
	
	Args:
		interaction_type: Type of web interaction (extract, search, analyze)
	"""
	def decorator(original_method: Callable) -> Callable:
		@functools.wraps(original_method)
		async def wrapper(*args, **kwargs) -> Any:
			parlant_service = get_parlant_service()
			
			# Build web interaction context
			context = {
				'interaction_type': interaction_type,
				'method_name': original_method.__name__,
				'args_count': len(args),
				'kwargs_keys': list(kwargs.keys())
			}
			
			# Extract page context if available
			browser_session = None
			for arg in args:
				if hasattr(arg, 'url') or hasattr(arg, 'cdp_url'):
					browser_session = arg
					break
			
			if browser_session:
				page_context = {
					'url': getattr(browser_session, 'current_url', None),
					'session_id': getattr(browser_session, 'id', None)
				}
			else:
				page_context = {}
			
			# Validate web interaction
			validation_result = await parlant_service.validate_web_interaction(
				interaction_type=interaction_type,
				target_info=context,
				page_context=page_context
			)
			
			if not validation_result['approved']:
				raise PermissionError(
					f"Parlant validation blocked {interaction_type} interaction: {validation_result['reasoning']}"
				)
			
			return await original_method(*args, **kwargs)
		
		return wrapper
	return decorator


def parlant_validated_session_management(session_action: str) -> Callable:
	"""
	Parlant validation decorator factory for session management
	
	Creates validation wrappers for browser session operations.
	
	Args:
		session_action: Type of session action (create, close, configure)
	"""
	def decorator(original_method: Callable) -> Callable:
		@functools.wraps(original_method)
		async def wrapper(*args, **kwargs) -> Any:
			parlant_service = get_parlant_service()
			
			# Build session management context
			context = {
				'session_action': session_action,
				'method_name': original_method.__name__,
				'args_count': len(args),
				'kwargs_keys': list(kwargs.keys())
			}
			
			# Extract session info if available
			session_info = {}
			if hasattr(args[0] if args else None, 'id'):
				session_info['session_id'] = args[0].id
			
			# Validate session management
			validation_result = await parlant_service.validate_session_management(
				session_action=session_action,
				session_info=session_info,
				browser_config=context
			)
			
			if not validation_result['approved']:
				raise PermissionError(
					f"Parlant validation blocked {session_action} session operation: {validation_result['reasoning']}"
				)
			
			return await original_method(*args, **kwargs)
		
		return wrapper
	return decorator


def enhance_agent_with_parlant(agent_class):
	"""
	Class decorator to enhance Agent classes with Parlant validation
	
	Wraps critical Agent methods with conversational AI validation.
	
	Args:
		agent_class: Agent class to enhance
		
	Returns:
		Enhanced agent class with Parlant validation
		
	Example:
		@enhance_agent_with_parlant
		class MyAgent(Agent):
			pass
	"""
	# Store original methods
	original_run = agent_class.run if hasattr(agent_class, 'run') else None
	original_step = agent_class.step if hasattr(agent_class, 'step') else None
	original_run_sync = agent_class.run_sync if hasattr(agent_class, 'run_sync') else None
	
	# Apply Parlant validation to key methods
	if original_run:
		agent_class.run = parlant_validated_agent_run(original_run)
	
	if original_step:
		agent_class.step = parlant_validated_agent_step(original_step)
	
	if original_run_sync:
		@functools.wraps(original_run_sync)
		def parlant_validated_run_sync(self, **kwargs):
			# For sync methods, run async validation in event loop
			try:
				loop = asyncio.get_event_loop()
			except RuntimeError:
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
			
			async def async_run_sync():
				parlant_service = get_parlant_service()
				
				context = {
					'method': 'agent_run_sync',
					'task_description': getattr(self, 'task', 'Unknown task'),
					'sync_execution': True
				}
				
				validation_result = await parlant_service.validate_task_execution(
					task_description=context['task_description'],
					execution_context=context
				)
				
				if not validation_result['approved']:
					raise PermissionError(
						f"Parlant validation blocked sync agent run: {validation_result['reasoning']}"
					)
				
				return original_run_sync(self, **kwargs)
			
			return loop.run_until_complete(async_run_sync())
		
		agent_class.run_sync = parlant_validated_run_sync
	
	return agent_class


def enhance_tools_with_parlant(tools_class):
	"""
	Class decorator to enhance Tools classes with Parlant validation
	
	Wraps all browser action methods with conversational AI validation.
	
	Args:
		tools_class: Tools class to enhance
		
	Returns:
		Enhanced tools class with Parlant validation
		
	Example:
		@enhance_tools_with_parlant
		class MyTools(Tools):
			pass
	"""
	# Define method mappings for different action types
	browser_actions = {
		'click_element_by_index': 'click',
		'input_text': 'type',
		'go_to_url': 'navigate',
		'upload_file_to_element': 'upload',
		'scroll': 'scroll',
		'send_keys': 'type',
		'switch_tab': 'navigate',
		'close_tab': 'navigate',
		'go_back': 'navigate'
	}
	
	web_interactions = {
		'extract_structured_data': 'extract',
		'extract_clean_markdown': 'extract',
		'scroll_to_text': 'search',
		'get_dropdown_options': 'extract',
		'select_dropdown_option': 'select',
		'search_google': 'search'
	}
	
	session_operations = {
		'switch_tab': 'switch',
		'close_tab': 'close'
	}
	
	# Apply validation to browser action methods
	for method_name, action_type in browser_actions.items():
		if hasattr(tools_class, method_name):
			original_method = getattr(tools_class, method_name)
			if asyncio.iscoroutinefunction(original_method):
				setattr(tools_class, method_name, parlant_validated_browser_action(action_type)(original_method))
	
	# Apply validation to web interaction methods
	for method_name, interaction_type in web_interactions.items():
		if hasattr(tools_class, method_name):
			original_method = getattr(tools_class, method_name)
			if asyncio.iscoroutinefunction(original_method):
				setattr(tools_class, method_name, parlant_validated_web_interaction(interaction_type)(original_method))
	
	# Apply validation to session management methods
	for method_name, session_action in session_operations.items():
		if hasattr(tools_class, method_name):
			original_method = getattr(tools_class, method_name)
			if asyncio.iscoroutinefunction(original_method):
				setattr(tools_class, method_name, parlant_validated_session_management(session_action)(original_method))
	
	return tools_class


# Validation decorators for individual functions
def parlant_validate_task_execution(func: Callable) -> Callable:
	"""Decorator for task execution functions"""
	@functools.wraps(func)
	async def wrapper(*args, **kwargs):
		parlant_service = get_parlant_service()
		
		context = {
			'function_name': func.__name__,
			'args': len(args),
			'kwargs': list(kwargs.keys())
		}
		
		validation_result = await parlant_service.validate_task_execution(
			task_description=f"Execute {func.__name__}",
			execution_context=context
		)
		
		if not validation_result['approved']:
			raise PermissionError(f"Parlant blocked {func.__name__}: {validation_result['reasoning']}")
		
		return await func(*args, **kwargs)
	
	return wrapper


def parlant_validate_browser_operation(func: Callable) -> Callable:
	"""Decorator for browser operation functions"""
	@functools.wraps(func)
	async def wrapper(*args, **kwargs):
		parlant_service = get_parlant_service()
		
		context = {
			'function_name': func.__name__,
			'args': len(args),
			'kwargs': list(kwargs.keys())
		}
		
		validation_result = await parlant_service.validate_browser_action(
			action=func.__name__,
			task_context=context
		)
		
		if not validation_result['approved']:
			raise PermissionError(f"Parlant blocked {func.__name__}: {validation_result['reasoning']}")
		
		return await func(*args, **kwargs)
	
	return wrapper


def parlant_validate_web_operation(func: Callable) -> Callable:
	"""Decorator for web operation functions"""
	@functools.wraps(func)
	async def wrapper(*args, **kwargs):
		parlant_service = get_parlant_service()
		
		context = {
			'function_name': func.__name__,
			'args': len(args),
			'kwargs': list(kwargs.keys())
		}
		
		validation_result = await parlant_service.validate_web_interaction(
			interaction_type=func.__name__,
			target_info=context
		)
		
		if not validation_result['approved']:
			raise PermissionError(f"Parlant blocked {func.__name__}: {validation_result['reasoning']}")
		
		return await func(*args, **kwargs)
	
	return wrapper


# Health and diagnostics
def get_parlant_validation_status() -> Dict[str, Any]:
	"""
	Get comprehensive status of Parlant validation for Browser-Use
	
	Returns:
		Detailed status including service health and metrics
	"""
	parlant_service = get_parlant_service()
	return {
		'service': 'browser-use',
		'parlant_integration': 'active',
		'validation_enabled': parlant_service.PARLANT_ENABLED,
		'health_status': parlant_service.get_health_status(),
		'timestamp': datetime.now().isoformat()
	}


# Export validation functions for direct use
__all__ = [
	'ParlantValidatedBrowserUse',
	'enhance_agent_with_parlant',
	'enhance_tools_with_parlant',
	'parlant_validate_task_execution',
	'parlant_validate_browser_operation',
	'parlant_validate_web_operation',
	'get_parlant_validation_status'
]