import express from 'express'
import mongoose from 'mongoose'
import modelsRouter from './routes/api/model.js'
import promptsRouter from './routes/api/prompts.js'
import runsRouter from './routes/api/runs.js'
import inferenceRouter from './routes/api/inference.js'

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/ablitmd'

if (process.env.NODE_ENV !== 'test') {
  mongoose.connect(MONGO_URI)
    .then(() => console.log('MongoDB connected'))
    .catch(err => console.error('MongoDB connection error:', err))
}

const app = express()

app.use(express.json())

app.use('/api/models', modelsRouter)
app.use('/api/prompts', promptsRouter)
app.use('/api/runs', runsRouter)
app.use('/api/inference', inferenceRouter)

app.use((err, req, res, next) => {
  console.error(err)
  res.status(500).json({ error: err.message || 'Internal server error' })
})

export default app
