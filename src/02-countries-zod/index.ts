import { z } from "zod";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from "langchain/prompts";
import { LLMChain } from "langchain/chains";
import {
  StructuredOutputParser,
  OutputFixingParser,
} from "langchain/output_parsers";

export const run = async () => {
  const outputParser = StructuredOutputParser.fromZodSchema(
    z
      .array(
        z.object({
          fields: z.object({
            Name: z.string().describe("国名"),
            Capital: z.string().describe("首都"),
          }),
        })
      )
      .describe("An array of Airtable records, each representing a country")
  );

  const chatModel = new ChatOpenAI({
    modelName: "gpt-4-1106-preview", // Or gpt-3.5-turbo
    temperature: 0, // For best results with the output fixing parser
  });

  const outputFixingParser = OutputFixingParser.fromLLM(
    chatModel,
    outputParser
  );

  // Don't forget to include formatting instructions in the prompt!
  const prompt = new PromptTemplate({
    template: `Answer the user's question as best you can:\n{format_instructions}\n{query}`,
    inputVariables: ["query"],
    partialVariables: {
      format_instructions: outputFixingParser.getFormatInstructions(),
    },
  });

  const answerFormattingChain = new LLMChain({
    llm: chatModel,
    prompt,
    outputKey: "records", // For readability - otherwise the chain output will default to a property named "text"
    outputParser: outputFixingParser,
  });

  const result = await answerFormattingChain.call({
    query: "8カ国ぶん教えてください。日本語名で回答してください",
  });

  console.log(JSON.stringify(result.records, null, 2));
};
/*
[
  {
    "fields": {
      "Name": "日本",
      "Capital": "東京"
    }
  },
  {
    "fields": {
      "Name": "アメリカ",
      "Capital": "ワシントンD.C."
    }
  },
  {
    "fields": {
      "Name": "イギリス",
      "Capital": "ロンドン"
    }
  },
  {
    "fields": {
      "Name": "フランス",
      "Capital": "パリ"
    }
  },
  {
    "fields": {
      "Name": "ドイツ",
      "Capital": "ベルリン"
    }
  },
  {
    "fields": {
      "Name": "中国",
      "Capital": "北京"
    }
  },
  {
    "fields": {
      "Name": "ロシア",
      "Capital": "モスクワ"
    }
  },
  {
    "fields": {
      "Name": "イタリア",
      "Capital": "ローマ"
    }
  }
]
*/
