import express from 'express'
import modelsRouter from './routes/api/model.js'
import promptsRouter from './routes/api/prompts.js'
import runsRouter from './routes/api/runs.js'

const app = express()
app.use(express.json())
app.use('/api/models', modelsRouter)
app.use('/api/prompts', promptsRouter)
app.use('/api/runs', runsRouter)

export default app
