import express from 'express'
import modelsRouter from './routes/api/model.js'
import promptsRouter from './routes/api/prompts.js'

const app = express()
app.use(express.json())
app.use('/api/models', modelsRouter)
app.use('/api/prompts', promptsRouter)

export default app
