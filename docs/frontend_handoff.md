# ML Engine: Frontend Integration Guide

This document outlines how to integrate the finalized Machine Learning Career Recommendation Engine into the frontend UI. 

## 1. The ML Artifacts
The `feat/production-export` branch provides two crucial static files for the frontend:
* **`riasec_model.onnx`**: The trained AI model that predicts the student's Top 3 broad career categories based on their quiz scores.
* **`riasec_jobs_db.json`**: The offline database of ~900 occupations, containing their specific RIASEC profiles and O*NET-SOC Codes.

## 2. The Data Contract
To power the "Reject & Replace" UI without needing a backend server, the frontend must execute the final math locally.
* **INPUT**: An array of 6 floats (0.0 to 1.0) representing the student's Holland Code quiz results: `[Realistic, Investigative, Artistic, Social, Enterprising, Conventional]`.
* **OUTPUT**: A JSON array of all applicable careers, sorted by highest `Match_Score`.

## 3. Implementation (JavaScript / Browser)
Because the app needs to run offline at the college fair, we recommend using the `onnxruntime-web` library to load the `.onnx` model directly in the browser. 

Once the ONNX model predicts the Top 3 categories, use the following JavaScript logic to filter and rank the `riasec_jobs_db.json` database.

```javascript
/**
 * Generates a ranked list of careers for the UI.
 * * @param {Array<number>} studentScores - [R, I, A, S, E, C]
 * @param {Array<string>} topCategories - The 3 strings predicted by the ONNX model
 * @param {Array<Object>} jobsDatabase - The parsed riasec_jobs_db.json
 * @returns {Array<Object>} - Sorted array of careers ready for display
 */
function getRecommendations(studentScores, topCategories, jobsDatabase) {
    // 1. Filter jobs to only the Top 3 categories
    const candidates = jobsDatabase.filter(job => 
        topCategories.includes(job['Career Category'])
    );

    // 2. Calculate Cosine Similarity for the remaining jobs
    const ranked = candidates.map(job => {
        const jobScores = [
            job.Realistic, job.Investigative, job.Artistic, 
            job.Social, job.Enterprising, job.Conventional
        ];
        const score = calculateCosineSimilarity(studentScores, jobScores);
        return { ...job, Match_Score: score };
    });

    // 3. Sort by highest Match_Score
    return ranked.sort((a, b) => b.Match_Score - a.Match_Score);
}

// Helper Function: Cosine Similarity Math
function calculateCosineSimilarity(vecA, vecB) {
    let dotProduct = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}