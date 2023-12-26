import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";

// .envの読み込み
require("dotenv").config();

// サンプル用の関数
export const make_index = async () => {
  // ✅ドキュメントの読み込み
  const loader = new PDFLoader("guideline.pdf");

  // ✅PDFファイルを500文字ごとに分割
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 500 });
  const docs = await loader.loadAndSplit(textSplitter);

  // ✅ドキュメントをベクトル化し、インデックスを生成
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  // ✅インデックスを保存
  await vectorStore.save("index"); // indexフォルダ
};

make_index();
