import React, {type ReactNode} from 'react';
import styles from './Callout.module.css';

type CalloutType = 'concept' | 'pitfall' | 'tip' | 'math';

interface Props {
  type: CalloutType;
  title?: string;
  children: ReactNode;
}

const config: Record<CalloutType, {icon: string; defaultTitle: string; color: string}> = {
  concept: {icon: '💡', defaultTitle: 'Key Concept', color: '#3b82f6'},
  pitfall: {icon: '⚠️', defaultTitle: 'Common Pitfall', color: '#f59e0b'},
  tip: {icon: '🚀', defaultTitle: 'Pro Tip', color: '#10b981'},
  math: {icon: '📐', defaultTitle: 'Math Note', color: '#8b5cf6'},
};

export default function Callout({type, title, children}: Props) {
  const {icon, defaultTitle, color} = config[type];

  return (
    <div className={styles.callout} style={{borderLeftColor: color}}>
      <div className={styles.header}>
        <span className={styles.icon}>{icon}</span>
        <strong>{title || defaultTitle}</strong>
      </div>
      <div className={styles.body}>{children}</div>
    </div>
  );
}
