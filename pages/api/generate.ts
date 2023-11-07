import { HfInference } from "@huggingface/inference";
import { HuggingFaceStream, StreamingTextResponse } from "ai";
import { experimental_buildOpenAssistantPrompt } from "ai/prompts";
import { NextApiRequest, NextApiResponse } from "next";

const Hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

export const runtime = "edge";

const enableCors = (req: NextApiRequest, res: NextApiResponse) => {
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Origin", "*"); // You may want to specify the allowed origins explicitly
  res.setHeader(
    "Access-Control-Allow-Methods",
    "GET,OPTIONS,PATCH,DELETE,POST,PUT"
  );
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Authorization, X-Requested-With, X-HTTP-Method-Override, Content-Type, Accept"
  );

  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }
};

export default async function POST(req: NextApiRequest, res: NextApiResponse) {
  enableCors(req, res);
  try {
    console.log("hello");
    // enableCors(req, res);

    const { messages } = req.body;
    console.log(req.body);
    if (!messages) {
      throw new Error("Request body does not contain messages.");
    }
    const response = await Hf.textGenerationStream({
      model:
        "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
      inputs: experimental_buildOpenAssistantPrompt(messages),
      parameters: {
        max_new_tokens: 200,
        // @ts-ignore (this is a valid parameter specifically in OpenAssistant models)
        typical_p: 0.2,
        repetition_penalty: 1,
        truncate: 1000,
        return_full_text: false,
      },
    });
    const stream = HuggingFaceStream(response);
    // Respond with the stream
    return new StreamingTextResponse(stream);
  } catch (e) {
    console.log("error");
  }
}
