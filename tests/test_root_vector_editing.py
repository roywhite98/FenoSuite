"""Domain transactions: edits must preserve ancestry and reject corruption."""
import unittest

from plant_phenotype.root_phenotype.vector.editing import apply_edit, _preserve_edits
from plant_phenotype.root_phenotype.vector.models import (
    BaseMark, ParentAttachment, RootDocument, RootNode, SourceRef, VectorRoot,
    validate_document,
)


def document_with_child():
    doc = RootDocument(source=SourceRef(width=100, height=100))
    doc.roots = [
        VectorRoot('main', [RootNode('a', 10, 10, 4), RootNode('b', 10, 40, 4), RootNode('c', 10, 80, 3)],
                   base_mark_id='base', direction='base_to_tip', review_state='auto', source_segment_ids=['s1']),
        VectorRoot('child', [RootNode('d', 10, 40, 2), RootNode('e', 50, 50, 1)],
                   parent_attachment=ParentAttachment('main', 'b'), direction='base_to_tip', review_state='auto', source_segment_ids=['s2']),
    ]
    doc.base_marks = [BaseMark('base', 10, 10, root_id='main')]
    return doc


class TestRootVectorEditing(unittest.TestCase):
    def test_parent_node_moves_child_origin_not_entire_child(self):
        original = document_with_child()
        result = apply_edit(original, dict(type='move_node', root_id='main', node_id='b', x=15, y=42))
        child = result.document.root_by_id('child')
        self.assertEqual((child.nodes[0].x, child.nodes[0].y), (15, 42))
        self.assertEqual((child.nodes[-1].x, child.nodes[-1].y), (50, 50))
        self.assertIsNone(child.nodes[0].diameter_px)
        self.assertIn('manual_locked', child.nodes[0].tags)
        self.assertEqual(original.root_by_id('main').nodes[1].x, 10)
        self.assertEqual(result.document.revision, original.revision + 1)

    def test_cannot_move_attached_base_away_from_parent(self):
        with self.assertRaises(ValueError):
            apply_edit(document_with_child(), dict(type='move_node', root_id='child', node_id='d', x=20, y=40))

    def test_cannot_reparent_primary_to_its_descendant(self):
        original = document_with_child()
        before = original.to_dict()
        with self.assertRaises(ValueError):
            apply_edit(original, dict(type='reparent', root_id='main', parent_root_id='child', parent_node_id='e'))
        self.assertEqual(before, original.to_dict())

    def test_detaching_does_not_create_a_primary(self):
        doc = apply_edit(document_with_child(), dict(type='detach', root_id='child')).document
        self.assertIsNone(doc.root_order('child'))
        self.assertEqual(doc.root_by_id('child').direction, 'unknown')
        self.assertEqual(doc.root_order('main'), 0)

    def test_delete_parent_keeps_child_unassigned_and_records_exclusion(self):
        doc = apply_edit(document_with_child(), dict(type='delete_root', root_id='main')).document
        self.assertEqual([r.id for r in doc.roots], ['child'])
        self.assertIsNone(doc.root_order('child'))
        self.assertIn('s1', doc.excluded_segment_ids)
        self.assertEqual(doc.base_marks, [])

    def test_delete_subtree_explicit(self):
        doc = apply_edit(document_with_child(), dict(type='delete_root', root_id='main', subtree=True)).document
        self.assertFalse(doc.roots)
        self.assertEqual(doc.excluded_segment_ids, ['s1', 's2'])

    def test_set_scale_and_manual_diameter(self):
        source = document_with_child()
        doc = apply_edit(source, dict(type='set_scale', mm_per_pixel=0.05)).document
        doc = apply_edit(doc, dict(type='set_diameter', root_id='main', node_id='b', diameter_px=5)).document
        self.assertEqual(doc.source.mm_per_pixel, 0.05)
        self.assertEqual(doc.config.mm_per_pixel, 0.05)
        self.assertEqual(doc.root_by_id('main').nodes[1].diameter_source, 'manual')
        self.assertIsNone(source.source.mm_per_pixel)

    def test_nonfinite_or_negative_measurements_rejected(self):
        for value in (float('nan'), float('inf'), 0, -3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_edit(document_with_child(), dict(type='set_diameter', root_id='main', node_id='b', diameter_px=value))

    def test_reverse_requires_detach_for_child(self):
        with self.assertRaises(ValueError):
            apply_edit(document_with_child(), dict(type='reverse', root_id='child'))
        doc = apply_edit(document_with_child(), dict(type='reverse', root_id='main')).document
        self.assertEqual(doc.root_by_id('main').nodes[0].id, 'c')
        self.assertEqual((doc.base_marks[0].x, doc.base_marks[0].y), (10, 80))
        self.assertEqual(doc.root_order('child'), 1)

    def test_invalid_ancestry_rejected(self):
        doc = document_with_child()
        doc.root_by_id('child').parent_attachment.parent_node_id = 'missing'
        with self.assertRaises(ValueError):
            validate_document(doc)

    def test_confirmed_identity_survives_unrelated_rebuild(self):
        old = apply_edit(document_with_child(), dict(type='confirm_root', root_id='child')).document
        rebuilt = document_with_child()
        _preserve_edits(old, rebuilt)
        self.assertEqual(rebuilt.root_by_id('child').review_state, 'confirmed')

    def test_changed_confirmed_identity_requires_review(self):
        old = apply_edit(document_with_child(), dict(type='confirm_root', root_id='child')).document
        rebuilt = document_with_child()
        rebuilt.root_by_id('child').source_segment_ids.append('s3')
        _preserve_edits(old, rebuilt)
        self.assertEqual(rebuilt.root_by_id('child').review_state, 'pending')

    def test_locked_parent_diameter_does_not_erase_child_diameter(self):
        old = apply_edit(document_with_child(), dict(type='set_diameter', root_id='main', node_id='b', diameter_px=5)).document
        rebuilt = document_with_child()
        _preserve_edits(old, rebuilt)
        self.assertEqual(rebuilt.root_by_id('child').nodes[0].diameter_px, 2)


if __name__ == '__main__':
    unittest.main()
