chunk_to_questions = """
You are given a table chunk with the following content:

{table_chunk}

### Your Task:
Generate **{questions_per_chunk} diverse questions** that would retrieve this specific table chunk. The questions should be based on the actual content shown in the table above.

### Question Types to Cover:
1. **Entity-specific query**: Ask about specific values, names, or entities that appear in the table
2. **Temporal query**: If time-related data exists, ask about specific time periods or dates
3. **Comparison/Ranking query**: Ask which entity has the most/least/highest/lowest value
4. **Aggregation query**: Ask about counts, averages, sums with specific conditions
5. **Complex reasoning query**: Combine multiple conditions or multi-step questions

### Important Requirements:
- Use **natural, conversational language** - vary the question structure
- Make questions **specific to the actual content** shown in the table, not generic
- Reference real values from the table when possible
- Questions should be answerable by looking at this table chunk
- **Language: {lang}** - Generate all questions in this language

### Output Format (JSON only):
```json
{{
    "questions": ["question1", "question2", "question3", ...]
}}
```

Generate {questions_per_chunk} questions now:
"""