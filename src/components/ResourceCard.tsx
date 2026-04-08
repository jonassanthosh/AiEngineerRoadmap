import React from 'react';
import styles from './ResourceCard.module.css';

type ResourceType = 'paper' | 'video' | 'tutorial' | 'tool' | 'book' | 'course';

interface Props {
  title: string;
  url: string;
  type: ResourceType;
  author?: string;
  description?: string;
}

const typeConfig: Record<ResourceType, {label: string; color: string}> = {
  paper: {label: 'Paper', color: '#8b5cf6'},
  video: {label: 'Video', color: '#ef4444'},
  tutorial: {label: 'Tutorial', color: '#3b82f6'},
  tool: {label: 'Tool', color: '#10b981'},
  book: {label: 'Book', color: '#f59e0b'},
  course: {label: 'Course', color: '#ec4899'},
};

export default function ResourceCard({title, url, type, author, description}: Props) {
  const {label, color} = typeConfig[type];

  return (
    <a href={url} target="_blank" rel="noopener noreferrer" className={styles.card}>
      <div className={styles.header}>
        <span className={styles.badge} style={{backgroundColor: color}}>
          {label}
        </span>
        <span className={styles.arrow}>↗</span>
      </div>
      <h4 className={styles.title}>{title}</h4>
      {author && <p className={styles.author}>by {author}</p>}
      {description && <p className={styles.description}>{description}</p>}
    </a>
  );
}
