import { OpenAI } from "langchain/llms/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { RetrievalQAChain } from "langchain/chains";

// .envの読み込み
require("dotenv").config();

// サンプル用の関数
export const run = async () => {
  // ✅作成済みのインデックスを読み込む
  const vectorStore = await HNSWLib.load(
    "index", // indexフォルダ
    new OpenAIEmbeddings()
  );

  // ✅モデル
  const model = new OpenAI({}); // OpenAIモデル
  // ✅チェーン
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  // ✅質問する
  const res = await chain.call({
    query:
      "現在の医療情報システムガイドラインのバージョンはいくつですか？",
  });

  console.log({ res });
};

run();

/*
{
    query: "医療情報システムのソフトウェアの構成管理について、注意すべき点を教えてください",
}
{
  res: {
    text: ' 医療情報システムのソフトウェアの構成管理において、適切な手順やバッチ処理の仕組みが整備されているか、本来構成すべきソフトウェアのバージョンや管理状況が確認されるべきです。また、クラウドサービスを利用する場合には、事業者による構成管理の手順の確認が必要です。'
  }
}

{
    query:
      "医療情報システムを作成するに当たって、AWSのようなクラウドサービスを利用することは可能ですか？　またその際の注意すべき点をオンプレミスと比較して教えてください",
  }
  {
  res: {
    text: ' クラウドサービスを利用することは可能ですが、医療機関等が定める安全管理の基準を満たさないサービスやプライバシーポリシーと整合性が取れないサービスを利用するリスクがあるため、事前に医療機関等が確認して安全性を確保する必要があります。また、クラウドサービスの場合、利用者と事業者の責任分界を明確に定め、具体的な管理内容を取り決める必要があります。オンプレミスと比較すると、クラウドサービスを利用する際には医療機関等がより細かな管理が必要になると言えます。'
  }
}

{
    query:
      "医療情報システムの認証認可についてチェックすべき点をリスト形式で教えてください",
  }
  {
  res: {
    text: ' クラウドサービスを利用することは可能ですが、医療機関等が定める安全管理の基準を満たさないサービスやプライバシーポリシーと整合性が取れないサービスを利用するリスクがあるため、事前に医療機関等が確認して安全性を確保する必要があります。また、クラウドサービスの場合、利用者と事業者の責任分界を明確に定め、具体的な管理内容を取り決める必要があります。オンプレミスと比較すると、クラウドサービスを利用する際には医療機関等がより細かな管理が必要になると言えます。'
  }
}
*/
