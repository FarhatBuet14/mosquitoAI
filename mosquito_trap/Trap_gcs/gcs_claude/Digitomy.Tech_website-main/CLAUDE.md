# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digitomy is a mosquito surveillance platform with two main parts: a React frontend (marketing site + annotation tool) and an Express/TypeScript backend API. The frontend is hosted on Firebase (`digitomy.tech`), and the backend deploys to Google Cloud Run with Cloud SQL (PostgreSQL).

## Common Commands

### Frontend (root directory)
- `npm start` — Start React dev server (port 3000)
- `npm run build` — Production build to `build/`
- `npm test` — Run tests (react-scripts/jest)
- `firebase deploy` — Deploy frontend (change `firebase.json` "site" between "digitomy-temp" and "digitomy" for staging vs production)

### Backend (`backend/` directory)
- `npm run dev` — Start dev server with hot reload (ts-node-dev, port 5000/8080)
- `npm run build` — Compile TypeScript to `dist/`
- `npm start` — Run compiled production server
- `npm run prisma:generate` — Generate Prisma client after schema changes
- `npm run prisma:migrate` — Create and run database migrations
- `npm run prisma:studio` — Open Prisma database GUI
- `npm run db:setup-local` — Reset local DB, regenerate client, and seed
- `npm run db:seed-local` — Seed local database with test data

## Architecture

### Frontend (React + Tailwind CSS)
- **Entry**: `src/App.js` — React Router with 4 routes: `/` (landing), `/auth`, `/annotate`, `/contact`
- **Pages**: `src/Screens/` — `LandingPage.js` composes section components; `AnnotatePage.jsx` is the annotation interface; `AuthPage.jsx` handles login/register
- **UI Components**: `src/Screens/Components/` — Reusable animated components (TSX, using framer-motion and tsparticles)
- **Sections**: `src/Screens/Sections/` — Landing page sections (Hero, About, TrapAnnotate, Recognitions, Demo, Team, Collaborators, Patents, Contact)
- **Styling**: Tailwind CSS + MUI; config in `tailwind.config.js` and `postcss.config.js`
- **Assets**: `src/Assets/` — Images and videos organized by section
- **API URL**: Set via `REACT_APP_API_URL` env var (see `env.example.txt`)

### Backend (Express + Prisma + PostgreSQL)
- **Entry**: `backend/src/server.ts` — Express app with CORS, routes, health check
- **Routes**: `backend/src/routes/` — auth, images, annotations, traps, mosquitoImages, species, contact
- **Controllers**: `backend/src/controllers/` — Business logic for each route group
- **Database**: Prisma ORM with schema at `backend/prisma/schema.prisma`
- **Auth**: JWT-based with middleware at `backend/src/middleware/auth.ts`
- **Key utility**: `backend/src/utils/consensus.ts` — Consensus algorithm (3 matching genus+species annotations marks an image as identified)
- **GCS integration**: `backend/src/utils/gcs.ts` — Google Cloud Storage for image serving

### Data Model (key entities)
- **Organization** — Multi-tenant isolation
- **User** — Belongs to organization, authenticates via JWT
- **Capture** — A trap capture event with metadata (place, species, DNA barcoded status)
- **RawImage / ProcessedImage / LocalizedImage / MosquitoImage** — Image pipeline stages, all linked to Capture
- **UserAnnotation** — User's species annotation on a MosquitoImage (unique per user+image)
- **TrapInfo** — Physical trap device metadata
- **ContactMessage** — Contact form submissions

### Deployment
- Frontend: Firebase Hosting (build then `firebase deploy`)
- Backend: Google Cloud Run via Docker (`backend/deploy.sh`, `backend/Dockerfile`)
- Database: Cloud SQL PostgreSQL (setup via `backend/setup-gcp.sh`)
- Migration scripts: `backend/migrate.sh` for Cloud SQL proxy migrations

## Environment Variables

### Frontend
- `REACT_APP_API_URL` — Backend API base URL

### Backend (in `backend/.env`)
- `DATABASE_URL` — PostgreSQL connection string
- `JWT_SECRET` — JWT signing secret
- `PORT` — Server port (default 8080)
- `NODE_ENV` — development/production
- `FRONTEND_URL` — For CORS allowlist

## Git Workflow

Never push commits directly to the `main` branch. Always create a feature branch and commit changes there.

## Mixed Language Note

The frontend mixes `.js`, `.jsx`, and `.tsx` files. Section/page components use JSX; reusable animated components in `Components/` use TSX. The backend is fully TypeScript.
