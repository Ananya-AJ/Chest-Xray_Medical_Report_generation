import streamlit as st
from PIL import Image
import asyncio
import openai

import tester, config
from chatbot import *

global_context = None
openai.api_key = config.openai_api_key

class GenerateReportChatbot():
    def __init__(self):
        self.inference = tester.Inference()

    async def chatbot_response(self, user_input,report):
        global global_context
        response = ''
        # await populate_memory(kernel, report)
        # await search_memory_examples(kernel,report)
        if global_context is not None:
                chat_func, context = await setup_chat_with_memory(kernel, report, global_context)
        else:
                chat_func, context = await setup_chat_with_memory(kernel, report)
            
        context["user_input"] = user_input
        context["report"] = report
        answer = await chat(kernel, chat_func, context, user_input)
        if answer:     
            response += str(answer)
    
        # save context for the next function call
        global_context = context
        
        return response

    async def chatbot_loop(self, report):
        count = 0
        user_input = st.text_input('You:', key=str(count))
        send_button = st.button('Send', key=f"button-{count}")
        await populate_memory(kernel, report)
        # await search_memory_examples(kernel,report)
        # chat_func, context = await setup_chat_with_memory(kernel, report)

        while True:
            if send_button:
                if user_input != "bye":
                    response = await self.chatbot_response(user_input, report)
                    st.text_area('Bot:', response)
                else:
                    st.text_area('Bot:', "Have a Great Day!")
                    break

                count += 1
                user_input = st.text_input('You:', key=str(count))
                send_button = st.button('Send', key=f"button-{count}")
            else:
                await asyncio.sleep(0.1)

    async def app(self):
        '''
        Launches streamlit app, takes input image and generates report which is displayed to user.
        '''
        # Set page title
        st.title('Medical Report Generation for Chest X-rays')

        # Add file uploader for image input
        uploaded_file = st.file_uploader('Choose an X-ray image', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded X-ray image', use_column_width=True)

        # Add button to generate report
        if st.button('Generate Report'):
            if uploaded_file is not None:
                # Get image_id of uploaded image
                image_id = uploaded_file.name

                # Generate report
                pred_tags_prob, pred_tags, pred_report = await self.inference.generate_report(image, image_id)
                # Format generated report to extract findings and impressions from it
                pred_report_split = pred_report.split(':')
                pred_impression = pred_report_split[3]
                pred_findings = pred_report_split[2]
                pred_findings_ = pred_findings.rsplit('.', 1)[0]

                # Calculate metrics
                hamming_loss, impressions_sim, findings_sim = tester.Inference().evaluate_metrics(image_id, pred_tags_prob, pred_impression, pred_findings_)

                # Display report and metrics to user
                st.success(f'Impression: {pred_impression}\n\nFindings: {pred_findings_}\n\nTags: {pred_tags}')

                st.success(f'Hamming loss: \n\nTags: {hamming_loss}')
                st.success(f'Semantic similarity: \n\nImpression: {impressions_sim}\n\nFindings: {findings_sim}')

                st.title('Talk to our Chatbot')                
                await self.chatbot_loop(pred_report)
            else:
                st.error('Please upload an X-ray image.')


if __name__ == '__main__':
    obj = GenerateReportChatbot()
    #obj.app()
    asyncio.run(obj.app())