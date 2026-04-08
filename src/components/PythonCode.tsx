import React, {type ReactNode} from 'react';
import CodeBlock from '@theme/CodeBlock';
import styles from './PythonCode.module.css';

interface Props {
  children: string;
  title?: string;
  colabUrl?: string;
  showLineNumbers?: boolean;
}

export default function PythonCode({children, title, colabUrl, showLineNumbers = true}: Props) {
  return (
    <div className={styles.wrapper}>
      <CodeBlock language="python" title={title} showLineNumbers={showLineNumbers}>
        {children}
      </CodeBlock>
      {colabUrl && (
        <a
          href={colabUrl}
          target="_blank"
          rel="noopener noreferrer"
          className={styles.colabButton}>
          <img
            src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open in Colab"
          />
        </a>
      )}
    </div>
  );
}
