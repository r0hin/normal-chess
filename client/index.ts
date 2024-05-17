import dotenv from "dotenv";
import { cert, initializeApp } from "firebase-admin/app"
import { getFirestore } from "firebase-admin/firestore";
dotenv.config();

const app = initializeApp({ // @ts-ignore
  credential: cert({
    projectId: "ana-ai0",
    privateKey: (process.env.FIREBASE_PRIVATE_KEY as string).replace(/\\n/gm, "\n"),
    clientEmail: "firebase-adminsdk-laahy@ana-ai0.iam.gserviceaccount.com"
  }),
});
const db = getFirestore(app);

