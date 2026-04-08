import React, {useState, useEffect} from 'react';
import styles from './ProgressTracker.module.css';

const MONTHS = [
  {id: 'month-1', label: 'Month 1', subtitle: 'AI & ML Foundations'},
  {id: 'month-2', label: 'Month 2', subtitle: 'Deep Learning'},
  {id: 'month-3', label: 'Month 3', subtitle: 'NLP & Transformers'},
  {id: 'month-4', label: 'Month 4', subtitle: 'Large Language Models'},
  {id: 'month-5', label: 'Month 5', subtitle: 'LLM Optimization'},
  {id: 'month-6', label: 'Month 6', subtitle: 'Building New Models'},
];

const STORAGE_KEY = 'ai-academy-progress';

export default function ProgressTracker() {
  const [progress, setProgress] = useState<Record<string, boolean>>({});

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) setProgress(JSON.parse(saved));
    } catch {}
  }, []);

  const toggle = (id: string) => {
    const next = {...progress, [id]: !progress[id]};
    setProgress(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  const completed = Object.values(progress).filter(Boolean).length;
  const pct = Math.round((completed / MONTHS.length) * 100);

  return (
    <div className={styles.tracker}>
      <div className={styles.header}>
        <h3 className={styles.heading}>Your Progress</h3>
        <span className={styles.pct}>{pct}%</span>
      </div>
      <div className={styles.bar}>
        <div className={styles.fill} style={{width: `${pct}%`}} />
      </div>
      <div className={styles.months}>
        {MONTHS.map((m) => (
          <button
            key={m.id}
            className={`${styles.month} ${progress[m.id] ? styles.done : ''}`}
            onClick={() => toggle(m.id)}>
            <span className={styles.check}>{progress[m.id] ? '✓' : '○'}</span>
            <span className={styles.label}>{m.label}</span>
            <span className={styles.subtitle}>{m.subtitle}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
