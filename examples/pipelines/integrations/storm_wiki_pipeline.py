"""
title: Storm Wiki Pipeline
author: zabirauf
date: 2024-07-22
version: 1.0
license: MIT
description: Research about a topic and create a wikipedia like content containing summary and information with varied perspective.
requirements: knowledge_storm
environment_variables: OPENAI_API_KEY, YOU_API_KEY, REGULAR_MODEL_NAME, SMART_MODEL_NAME
"""

import tempfile
import os

from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.rm import YouRM
from pydantic import BaseModel, Field

class Tools:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(
            default="",
            description="The API key for OpenAI services"
        )
        YOU_API_KEY: str = Field(
            default="",
            description="The API key for You.com services"
        )
        REGULAR_MODEL_NAME: str = Field(
            default="gpt-4o-mini",
            description="The name of the regular language model to use"
        )
        SMART_MODEL_NAME: str = Field(
            default="gpt-4o",
            description="The name of the smart language model to use"
        )

    def __init__(self, valves) -> None:
        self.valves = valves
        self.lm_configs = STORMWikiLMConfigs()
        self.openai_kwargs = {
            'api_key': self.valves.OPENAI_API_KEY,
            'temperature': 1.0,
            'top_p': 0.9,
        }
        regular_model_name = self.valves.REGULAR_MODEL_NAME
        smart_model_name = self.valves.SMART_MODEL_NAME
        conv_simulator_lm = ModelClass(model=regular_model_name, max_tokens=500, **openai_kwargs)
        question_asker_lm = ModelClass(model=regular_model_name, max_tokens=500, **openai_kwargs)
        outline_gen_lm = ModelClass(model=smart_model_name, max_tokens=400, **openai_kwargs)
        article_gen_lm = ModelClass(model=smart_model_name, max_tokens=700, **openai_kwargs)
        article_polish_lm = ModelClass(model=smart_model_name, max_tokens=4000, **openai_kwargs)

        self.lm_configs.set_conv_simulator_lm(conv_simulator_lm)
        self.lm_configs.set_question_asker_lm(question_asker_lm)
        self.lm_configs.set_outline_gen_lm(outline_gen_lm)
        self.lm_configs.set_article_gen_lm(article_gen_lm)
        self.lm_configs.set_article_polish_lm(article_polish_lm)

        # Create a temporary directory for output
        self.temp_output_dir = tempfile.mkdtemp()
        print(f"Created temporary output directory: {self.temp_output_dir}")

        # Update engine_args with the temporary directory
        self.engine_args = STORMWikiRunnerArguments(
            output_dir=self.temp_output_dir,
            max_conv_turn=3,
            max_perspective=3,
            search_top_k=3,
            max_thread_num=3,
        )

        self.research_module = YouRM(ydc_api_key=self.valves.YOU_API_KEY, k=5)

    def research_topic(
        self,
        topic: str,
    ) -> str:
        """
        Research about a topic and create a wikipedia like content containing summary and information with varied perspective.

        :param topic: The topic user want to research about
        :return: The wikipedia like content containing summary and more detailed information about the topic
        """
        if self.valves.OPENAI_API_KEY == "":
            return "OpenAPI Key not set, ask the user to set it up."
        elif self.valves.YOU_API_KEY == "":
            return "You.com API Key not set, ask the user to set it up."
        else:
            runner = STORMWikiRunner(self.engine_args, self.lm_configs, self.research_module)
            runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
            )
            runner.post_run()
            runner.summary()

            # Read the polished article from the output directory
            article_path = os.path.join(runner.article_output_dir, 'storm_gen_article_polished.txt')
            if os.path.exists(article_path):
                with open(article_path, 'r', encoding='utf-8') as f:
                    polished_article = f.read()
                return polished_article
            else:
                return "Error: Polished article not found."