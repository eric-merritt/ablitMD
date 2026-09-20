import { CardStyle } from '../atoms/CardStyle'
import { Button } from '../atoms/Button'
import { ModalTitle } from './ModalTitle'
import { ModalBody } from './ModalBody'

export const ReclassifyModal = ({ onConfirm, onCancel }: { onConfirm: () => void; onCancel: () => void }) => (
  <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50 }} onClick={e => e.target === e.currentTarget && onCancel()}>
    <div style={{ ...CardStyle, width: '340px', boxShadow: '0 12px 40px rgba(0,0,0,.5)' }}>
      <ModalTitle />
      <ModalBody />
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
        <Button label="Cancel" onClick={onCancel} variant="ghost" />
        <Button label="Yes, reclassify" onClick={onConfirm} />
      </div>
    </div>
  </div>
)
