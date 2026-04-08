import React, {useState, type ReactNode} from 'react';
import styles from './ExerciseBlock.module.css';

type Difficulty = 'beginner' | 'intermediate' | 'advanced';

interface Props {
  title: string;
  difficulty: Difficulty;
  children: ReactNode;
  hints?: string[];
  solution?: ReactNode;
}

const difficultyColors: Record<Difficulty, string> = {
  beginner: '#22c55e',
  intermediate: '#f59e0b',
  advanced: '#ef4444',
};

export default function ExerciseBlock({title, difficulty, children, hints, solution}: Props) {
  const [showHints, setShowHints] = useState(false);
  const [showSolution, setShowSolution] = useState(false);

  return (
    <div className={styles.exercise}>
      <div className={styles.header}>
        <span className={styles.icon}>✏️</span>
        <span className={styles.title}>{title}</span>
        <span
          className={styles.badge}
          style={{backgroundColor: difficultyColors[difficulty]}}>
          {difficulty}
        </span>
      </div>
      <div className={styles.body}>{children}</div>
      {hints && hints.length > 0 && (
        <div className={styles.section}>
          <button
            className={styles.toggle}
            onClick={() => setShowHints(!showHints)}>
            {showHints ? '▼' : '▶'} Hints ({hints.length})
          </button>
          {showHints && (
            <ol className={styles.hints}>
              {hints.map((hint, i) => (
                <li key={i}>{hint}</li>
              ))}
            </ol>
          )}
        </div>
      )}
      {solution && (
        <div className={styles.section}>
          <button
            className={styles.toggle}
            onClick={() => setShowSolution(!showSolution)}>
            {showSolution ? '▼' : '▶'} Solution
          </button>
          {showSolution && <div className={styles.solution}>{solution}</div>}
        </div>
      )}
    </div>
  );
}
