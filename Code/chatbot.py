from typing import Tuple, Optional
import semantic_kernel as sk
from semantic_kernel.kernel_config import KernelConfig
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion, OpenAITextEmbedding

import config

api_key = config.openai_api_key

# Semantic kernel settings
kernel = sk.Kernel()
kernel.config.add_text_completion_service('dv', OpenAITextCompletion('text-davinci-003', api_key))
kernel.config.add_text_embedding_generation_service('ada', OpenAITextEmbedding('text-embedding-ada-002', api_key))
kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
kernel.import_skill(sk.core_skills.TextMemorySkill())

async def populate_memory(kernel, report) -> None:
    '''
    Add some documents to the semantic memory

    Args:
    kernel(Semantic Kernel): an instance of Semantic Kernel
    report(str): predicted report string 
    '''
    await kernel.memory.save_information_async(
        'report', id='cause', text=report
    )
    await kernel.memory.save_information_async(
        'report', id='prevention', text=report
    )
    await kernel.memory.save_information_async(
        'report', id='symptoms', text=report
    )
    await kernel.memory.save_information_async(
        'report', id='explain report', text=report
    )
    await kernel.memory.save_information_async(
        'report', id='seriousness of report indication', text=report
    )

async def setup_chat_with_memory(kernel, report, context=None) -> Tuple[sk.SKFunctionBase, sk.SKContext]:
    '''
    Setup chat with memory in real time based on the conversation between bot and user

    Args:
    kernel(Semantic Kernel): an instance of Semantic Kernel
    report(str): predicted report string 
    context(Semantic Kernel context): semantic kernel context

    Returns:
    chat_func(Semantic Kernel function base): Semantic kernel function base
    context(Semantic Kernel context): semantic kernel context
    '''

    sk_prompt = '''
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say "I don't know" if
    it does not have an answer.

    Information about report, from previous conversations:
    - {{$fact1}} {{recall $fact1}}
    - {{$fact2}} {{recall $fact2}}
    - {{$fact3}} {{recall $fact3}}
    - {{$fact4}} {{recall $fact4}}
    - {{$fact5}} {{recall $fact5}}

    Chat:
    {{$chat_history}}
    User: {{$user_input}}
    ChatBot: '''.strip()

    chat_func = kernel.create_semantic_function(sk_prompt, max_tokens=200, temperature=0.2)

    if context is None:
        context = kernel.create_new_context()
        context['fact1'] = 'what are the symptoms'
        context['fact2'] = 'what are causes'
        context['fact3'] = 'How can I prevent?'
        context['fact4'] = 'What does my report say?'
        context['fact5'] = 'Does my report show a serious concern?'

        context[sk.core_skills.TextMemorySkill.COLLECTION_PARAM] = 'report'
        context[sk.core_skills.TextMemorySkill.RELEVANCE_PARAM] = 0.8

        context['chat_history'] = ''
        context['report'] = report

    for i, question in enumerate(['what are the symptoms', 'what are causes', 'How can I prevent','What does my report say', 'Does my report show a serious concern']):
        result = await kernel.memory.search_async('report', question)
        if len(result) >= 1:
            context[f'fact{i+1}'] = result[0].text

    return chat_func, context

async def chat(kernel, chat_func, context, user_input):
    '''
    Actual chat function whcih takes user input/ part of conversation and returns a response along with setting the context

    Args:
    kernel(Semantic Kernel): an instance of Semantic Kernel
    chat_func(Semantic Kernel function base): Semantic kernel function base
    context(Semantic Kernel context): semantic kernel context
    user_input(str): user input from chat

    Returns:
    answer(bool): Success in setting context 
    '''
    answer = await kernel.run_async(chat_func, input_vars=context.variables)
    context['chat_history'] += f'\nUser:> {user_input}\nChatBot:> {answer}\n'
    return answer

async def chatbot():
    await populate_memory(kernel)
    chat_func, context = await setup_chat_with_memory(kernel)
    await chatbot()